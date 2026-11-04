"""Evaluation script for FunctionGemma on held-out function-calling benchmarks.

Multi-Tool Aware evaluation with exact argument matching
across English, Burmese, and Code-Switching queries.
"""

import os
import re
import json
import random
import argparse
from typing import List, Dict, Any
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# ============================================================
# Register Classification
# ============================================================
BURMESE_RANGES = [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)]

def has_burmese(s: str) -> bool:
    return any(any(lo <= ord(c) <= hi for lo, hi in BURMESE_RANGES) for c in s)

def has_latin(s: str) -> bool:
    return any(65 <= ord(c) <= 90 or 97 <= ord(c) <= 122 for c in s)

def classify_register(text: str) -> str:
    b, e = has_burmese(text), has_latin(text)
    if b and e: return "Code-Switch"
    if b: return "Burmese"
    if e: return "English"
    return "Other"

def get_last_user_text(msgs: List[Dict[str, Any]]) -> str:
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else str(c)
    return ""

# ============================================================
# Function Calling Parsing & Argument Comparison
# ============================================================
def parse_fm_args(s: str) -> Dict[str, Any]:
    args = {}
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in ', \n\t': i += 1
        if i >= n: break
        c = s.find(':', i)
        if c < 0: break
        key = s[i:c].strip(); i = c + 1
        if s.startswith('<escape>', i):
            e = s.find('<escape>', i + 8)
            if e < 0: args[key] = s[i + 8:]; i = n
            else: args[key] = s[i + 8:e]; i = e + 8
        else:
            e = s.find(',', i)
            if e < 0: e = n
            raw = s[i:e].strip(); i = e
            if raw in ('None', ''): args[key] = None
            elif raw in ('True', 'true'): args[key] = True
            elif raw in ('False', 'false'): args[key] = False
            else:
                try: args[key] = float(raw) if '.' in raw else int(raw)
                except Exception: args[key] = raw
    return args

def extract_all_tool_calls(text: str) -> List[Dict[str, Any]]:
    if not text or not text.strip(): return []
    return [
        {"name": m.group(1), "arguments": parse_fm_args(m.group(2))}
        for m in re.finditer(r'<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>', text, re.DOTALL)
    ]

def clean_args(d: Any) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None} if isinstance(d, dict) else {}

def val_eq(val1: Any, val2: Any) -> bool:
    s1, s2 = str(val1).strip().lower(), str(val2).strip().lower()
    if s1 == s2: return True
    try: return float(s1) == float(s2)
    except Exception: return False

def compare_args(gt: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    g = clean_args(gt); p = clean_args(pred)
    if set(g.keys()) != set(p.keys()): return False
    return all(val_eq(p[k], g[k]) for k in g)

# ============================================================
# Main Evaluation Function
# ============================================================
def evaluate(
    model: Any,
    tokenizer: Any,
    eval_msgs: List[Any],
    global_tools: List[Dict[str, Any]],
    max_samples: int = 200,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    tools_by_name = {t["function"]["name"]: t for t in global_tools}
    parsed = [json.loads(m) if isinstance(m, str) else (m.tolist() if hasattr(m, 'tolist') else m) for m in eval_msgs]
    registers = [classify_register(get_last_user_text(m)) for m in parsed]

    random.seed(seed)
    by_reg = defaultdict(list)
    for i, r in enumerate(registers): by_reg[r].append(i)

    sample_indices = []
    if max_samples > 0 and max_samples < len(registers):
        for r, idxs in by_reg.items():
            n_take = max(1, round(max_samples * len(idxs) / len(registers)))
            sample_indices.extend(random.sample(idxs, min(n_take, len(idxs))))
        sample_indices = sample_indices[:max_samples]
        random.shuffle(sample_indices)
    else:
        sample_indices = list(range(len(registers)))

    results = defaultdict(lambda: {"total": 0, "exact": 0, "wrong_args": 0, "wrong_tool": 0})
    overall = {"total": 0, "exact": 0, "wrong_args": 0, "wrong_tool": 0}
    per_tool = defaultdict(lambda: {"total": 0, "ok": 0})

    model.eval()
    for idx in tqdm(sample_indices, desc="Evaluating Benchmark"):
        msgs = parsed[idx]; reg = registers[idx]
        gt_calls = []
        for m in msgs:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]: gt_calls.append(tc.get("function", {}))
        if not gt_calls: continue
        gt_names = [c["name"] for c in gt_calls]

        seen = set(); sample_tools = []
        for nm in gt_names:
            if nm in tools_by_name and nm not in seen:
                sample_tools.append(tools_by_name[nm]); seen.add(nm)
        sample_tools.extend([x for kk, x in tools_by_name.items() if kk not in seen][:2])

        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        formatted = tokenizer.apply_chat_template(
            prompt_msgs, tools=sample_tools, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2800).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        pred_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        pred_calls = extract_all_tool_calls(pred_text)

        if not pred_calls:
            cat = "wrong_tool"
        else:
            pred_names = [c["name"] for c in pred_calls]
            if set(pred_names) != set(gt_names):
                cat = "wrong_tool"
            else:
                all_ok = all(
                    next((compare_args(gt["arguments"], p["arguments"]) for p in pred_calls if p["name"] == gt["name"]), False)
                    for gt in gt_calls
                )
                cat = "exact" if all_ok else "wrong_args"

        results[reg]["total"] += 1; overall["total"] += 1
        results[reg][cat] += 1; overall[cat] += 1

        for gt in gt_calls:
            nm = gt["name"]; per_tool[nm]["total"] += 1
            match = next((p for p in pred_calls if p["name"] == nm), None)
            if match and compare_args(gt["arguments"], match["arguments"]):
                per_tool[nm]["ok"] += 1

    def pct(n: int, d: int) -> float: return (n / d * 100) if d else 0.0

    print("\n" + "=" * 70)
    print(f"Evaluation Benchmark (Multi-Tool Aware | {overall['total']} samples)")
    print("=" * 70)
    print(f"{'Register':<14}{'Samples':>8}   {'Fn Acc':>8}   {'Arg Acc':>9}   {'EM':>7}")
    print("-" * 70)
    for cat_name in ["English", "Burmese", "Code-Switch"]:
        r = results[cat_name]
        if r["total"] == 0: continue
        fn_ok = r["exact"] + r["wrong_args"]
        print(f"{cat_name:<14}{r['total']:>8}   {pct(fn_ok, r['total']):>7.1f}%   {pct(r['exact'], r['total']):>8.1f}%   {pct(r['exact'], r['total']):>6.1f}%")
    print("-" * 70)
    fn_all = overall["exact"] + overall["wrong_args"]
    print(f"{'Overall':<14}{overall['total']:>8}   {pct(fn_all, overall['total']):>7.1f}%   {pct(overall['exact'], overall['total']):>8.1f}%   {pct(overall['exact'], overall['total']):>6.1f}%\n")

    print("Per-Tool Accuracy (min 3 samples):")
    tool_stats = [(nm, s["ok"], s["total"], pct(s["ok"], s["total"])) for nm, s in per_tool.items() if s["total"] >= 3]
    tool_stats.sort(key=lambda x: x[3], reverse=True)
    print(f"{'Tool Name':<32}{'Correct':>9}{'Total':>7}{'Acc':>8}")
    print("-" * 58)
    for nm, ok, tot, acc in tool_stats:
        print(f"{nm:<32}{ok:>9}{tot:>7}{acc:>7.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FunctionGemma on held-out benchmarks")
    parser.add_argument("--model_path", type=str, default="thantzinphyo/Hey-270M-Alpha")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32).to(args.device)

    data_file = args.data_path
    if not data_file or not os.path.exists(data_file):
        for candidate in ["train.parquet", "../train.parquet"]:
            if os.path.exists(candidate):
                data_file = candidate; break
        else:
            print("Downloading 'train.parquet' from Hugging Face Hub (thantzinphyo/Hey-Function-Burmese)...")
            data_file = hf_hub_download(
                repo_id="thantzinphyo/Hey-Function-Burmese",
                filename="train.parquet",
                repo_type="dataset",
                local_dir=".",
            )

    tbl = pq.read_table(data_file, columns=["tools", "messages", "metadata"])
    global_tools = tbl.column("tools")[0].as_py()
    meta_col = tbl.column("metadata").to_pylist()
    msgs_col = tbl.column("messages").to_pylist()
    eval_msgs = [m for m, meta in zip(msgs_col, meta_col) if meta == "eval"]

    evaluate(model, tokenizer, eval_msgs, global_tools, max_samples=args.samples, seed=args.seed, device=args.device)

if __name__ == "__main__":
    main()
