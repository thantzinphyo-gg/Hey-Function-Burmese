"""Evaluation script for FunctionGemma on held-out function-calling benchmarks.

Parses function calls from model generations and reports:
- Tool Selection Accuracy
- Argument Matching Accuracy
- Exact Match (Overall)
across English, Burmese, and Code-Switching queries.
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any

import torch
from tqdm import tqdm
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download


def parse_function_calls(output_str: str) -> List[Dict[str, Any]]:
    """Extracts function names and JSON arguments from model output."""
    calls = []
    pattern = r"<start_function_call>call:([a-zA-Z0-9_]+)\s*(\{.*?\})<end_function_call>"
    for name, args_str in re.findall(pattern, output_str, re.DOTALL):
        clean = args_str.replace("<escape>", '"')
        quoted = re.sub(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', clean)
        try:
            args = json.loads(quoted)
        except Exception:
            kv = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*("(?:\\.|[^"\\])*"|[^,}]+)', clean)
            args = {k: v.strip('"') for k, v in kv}
        calls.append({"name": name, "arguments": args})
    return calls


def detect_language(text: str) -> str:
    """Classifies query into English, Burmese, or Code-Switching."""
    has_my = bool(re.search(r"[\u1000-\u109F]", text))
    has_en = bool(re.search(r"[a-zA-Z]", text))
    if has_my and has_en:
        return "Code-Switch"
    elif has_my:
        return "Burmese"
    return "English"


def evaluate(
    model: Any,
    tokenizer: Any,
    eval_msgs: List[Any],
    global_tools: List[Dict[str, Any]],
    max_samples: int = 200,
    device: str = "cuda",
) -> Dict[str, List[Dict[str, int]]]:
    tools_by_name = {t["function"]["name"]: t for t in global_tools}
    samples = eval_msgs[:max_samples] if max_samples > 0 else eval_msgs
    results = {"English": [], "Burmese": [], "Code-Switch": []}

    model.eval()
    for msgs in tqdm(samples, desc="Evaluating"):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        user_prompt = ""
        target_calls = []
        for m in msgs:
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    target_calls.append(tc.get("function", {}))

        lang = detect_language(user_prompt)
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]

        called = [t.get("name") for t in target_calls if t.get("name")]
        sample_tools = [tools_by_name[n] for n in called if n in tools_by_name]
        if not sample_tools:
            sample_tools = [global_tools[0]]

        formatted = tokenizer.apply_chat_template(
            prompt_msgs, tools=sample_tools, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1536).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        pred_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        pred_calls = parse_function_calls(pred_text)

        fn_ok = len(pred_calls) == len(target_calls) and all(
            p["name"] == t["name"] for p, t in zip(pred_calls, target_calls)
        )
        arg_ok = fn_ok and all(
            p["arguments"] == t["arguments"] for p, t in zip(pred_calls, target_calls)
        )

        results[lang].append({
            "tool_acc": 1 if fn_ok else 0,
            "arg_acc": 1 if arg_ok else 0,
            "exact_match": 1 if (fn_ok and arg_ok) else 0,
        })

    return results


def print_table(results: Dict[str, List[Dict[str, int]]]) -> None:
    print("\n" + "=" * 80)
    print(f"| {'Language Register':<18} | {'Samples':<8} | {'Tool Acc (%)':<14} | {'Arg Acc (%)':<14} | {'Exact Match (%)':<16} |")
    print("|:-------------------|:---------|:---------------|:---------------|:-----------------|")

    tot_n = tot_t = tot_a = tot_e = 0
    for lang, stats in results.items():
        if not stats:
            continue
        n = len(stats)
        t = sum(s["tool_acc"] for s in stats) / n * 100
        a = sum(s["arg_acc"] for s in stats) / n * 100
        e = sum(s["exact_match"] for s in stats) / n * 100
        tot_n += n
        tot_t += sum(s["tool_acc"] for s in stats)
        tot_a += sum(s["arg_acc"] for s in stats)
        tot_e += sum(s["exact_match"] for s in stats)
        print(f"| {lang:<18} | {n:<8} | {t:<14.1f} | {a:<14.1f} | {e:<16.1f} |")

    if tot_n > 0:
        print(f"| {'OVERALL':<18} | {tot_n:<8} | {tot_t/tot_n*100:<14.1f} | {tot_a/tot_n*100:<14.1f} | {tot_e/tot_n*100:<16.1f} |")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FunctionGemma model")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF ID of fine-tuned model")
    parser.add_argument("--data_path", type=str, default="train.parquet")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)

    tbl = pq.read_table(args.data_path, columns=["tools", "messages", "metadata"])
    global_tools = tbl.column("tools")[0].as_py()
    meta_col = tbl.column("metadata").to_pylist()
    msgs_col = tbl.column("messages").to_pylist()

    eval_msgs = [m for m, meta in zip(msgs_col, meta_col) if meta == "eval"]
    results = evaluate(model, tokenizer, eval_msgs, global_tools, max_samples=args.samples, device=device)
    print_table(results)


if __name__ == "__main__":
    main()
