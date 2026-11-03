"""Fine-tuning pipeline for FunctionGemma on multilingual tool calling.

Supports single and multi-GPU training with FP16 mixed precision,
chunked cross-entropy for low VRAM consumption, and dynamic collation.
"""

import os
import gc
import re
import math
import json
import time
import argparse
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from huggingface_hub import hf_hub_download


class FunctionGemmaDataset(Dataset):
    """Dynamic dataset with candidate tool filtering for memory-efficient training."""

    def __init__(
        self,
        messages_list: List[Any],
        global_tools: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int = 1536,
        include_distractors: int = 0,
    ):
        self.messages_list = messages_list
        self.tools_by_name = {t["function"]["name"]: t for t in global_tools}
        self.global_tools = global_tools
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.include_distractors = include_distractors
        self.resp_tokens = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.messages_list)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        msgs = self.messages_list[idx]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        elif hasattr(msgs, "tolist"):
            msgs = msgs.tolist()

        called = []
        for m in msgs:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    name = tc.get("function", {}).get("name")
                    if name:
                        called.append(name)

        sample_tools = [self.tools_by_name[n] for n in called if n in self.tools_by_name]
        if self.include_distractors > 0:
            distractors = [t for n, t in self.tools_by_name.items() if n not in called]
            sample_tools.extend(distractors[: self.include_distractors])

        if not sample_tools:
            sample_tools = [self.global_tools[0]]

        text = self.tokenizer.apply_chat_template(msgs, tools=sample_tools, tokenize=False)
        input_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        labels = list(input_ids)
        r_len = len(self.resp_tokens)
        match_idx = -1
        for i in range(len(input_ids) - r_len + 1):
            if input_ids[i : i + r_len] == self.resp_tokens:
                match_idx = i + r_len
                break

        if match_idx != -1:
            labels[:match_idx] = [-100] * match_idx
        else:
            labels = [-100] * len(labels)

        return {"input_ids": input_ids, "labels": labels}


def causal_collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """Dynamic padding collator guaranteeing shape alignment across devices."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []

    for x in batch:
        inp = x["input_ids"]
        lbl = x["labels"]
        pad_len = max_len - len(inp)

        input_ids.append(inp + [0] * pad_len)
        labels.append(lbl + [-100] * pad_len)
        attention_mask.append([1] * len(inp) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Computes cross-entropy loss sequentially over chunks to prevent OOM spikes."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)

    total_tokens = (flat_labels != ignore_index).sum()
    if total_tokens == 0:
        return flat_logits.sum() * 0.0

    accum_loss = 0.0
    for i in range(0, flat_labels.size(0), chunk_size):
        c_lbl = flat_labels[i : i + chunk_size]
        mask = c_lbl != ignore_index
        if mask.any():
            loss_chunk = torch.nn.functional.cross_entropy(
                flat_logits[i : i + chunk_size].float(),
                c_lbl,
                ignore_index=ignore_index,
                reduction="sum",
            )
            accum_loss = accum_loss + loss_chunk

    return accum_loss / total_tokens


def resolve_dataset(data_path: Optional[str], hf_token: Optional[str]) -> str:
    """Finds local dataset file or downloads from Hugging Face Hub."""
    if data_path and os.path.exists(data_path):
        return data_path

    for candidate in ["train.parquet", "../train.parquet"]:
        if os.path.exists(candidate):
            return candidate

    print("Downloading 'train.parquet' from Hugging Face Hub...")
    return hf_hub_download(
        repo_id="thantzinphyo/Gemma-Actions-MM",
        filename="train.parquet",
        repo_type="dataset",
        token=hf_token,
        local_dir=".",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma on Gemma-Actions-MM")
    parser.add_argument("--model_id", type=str, default="google/functiongemma-270m-it")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output_model")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", None))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--distractors", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Dataset
    resolved_path = resolve_dataset(args.data_path, args.hf_token)
    schema_table = pq.read_table(resolved_path, columns=["tools"])
    global_tools = schema_table.column("tools")[0].as_py()
    del schema_table; gc.collect()

    data_table = pq.read_table(resolved_path, columns=["messages", "metadata"])
    msgs_col = data_table.column("messages").to_pylist()
    meta_col = data_table.column("metadata").to_pylist()
    del data_table; gc.collect()

    train_msgs = [m for m, meta in zip(msgs_col, meta_col) if meta != "eval"]
    eval_msgs = [m for m, meta in zip(msgs_col, meta_col) if meta == "eval"]
    del msgs_col, meta_col; gc.collect()

    print(f"[Data] Train: {len(train_msgs):,} | Eval: {len(eval_msgs):,} | Tools: {len(global_tools)}")

    # 2. Tokenizer & Loader
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    train_dataset = FunctionGemmaDataset(
        train_msgs,
        global_tools,
        tokenizer,
        max_seq_length=args.max_seq_length,
        include_distractors=args.distractors,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=causal_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # 3. Model Initialization
    print(f"[Model] Loading '{args.model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4. Optimization Setup
    total_steps = math.ceil(len(train_msgs) / (args.batch_size * args.grad_accum)) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=36, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"[Training] Steps: {total_steps:,} | Effective Batch: {args.batch_size * args.grad_accum}")
    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    accum_loss = 0.0
    global_step = 0
    sub_step = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            sub_step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = chunked_cross_entropy(outputs.logits, labels, chunk_size=256, ignore_index=-100)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum

            if sub_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
                elapsed = time.time() - start_time
                sec_per_step = elapsed / max(global_step, 1)
                eta_sec = (total_steps - global_step) * sec_per_step
                current_loss = accum_loss / args.grad_accum
                accum_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]

                vram_info = ""
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
                    vram_info = f" | VRAM: {vram_gb:.2f} GB"

                print(
                    f"[Step {global_step:4d}/{total_steps:,}] "
                    f"Loss: {current_loss:.4f} | LR: {current_lr:.2e} | "
                    f"{sec_per_step:.1f}s/step (ETA: {int(eta_sec//3600)}h {int(eta_sec%3600//60):02d}m)"
                    f"{vram_info}",
                    flush=True,
                )

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    # 5. Save Output
    print(f"\n[Saving] Writing fine-tuned artifacts to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SUCCESS: Model and tokenizer saved cleanly.")


if __name__ == "__main__":
    main()
