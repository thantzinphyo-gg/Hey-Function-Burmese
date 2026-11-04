"""Fine-tuning pipeline for FunctionGemma on the Gemma-Actions-MM dataset.

Uses Hugging Face Trainer with CleanCausalTrainer architecture:
- Target function candidate extraction (~1,000 avg tokens)
- Dynamic batch padding with causal collation
- Zero-copy chunked cross-entropy (prevents 8GB VRAM allocation spike on 256K vocab)
- FP16 mixed-precision training with gradient checkpointing
- Real-time progress monitoring callback
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
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from huggingface_hub import hf_hub_download

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"


# ==============================================================================
# [1/4] Dataset & Collation
# ==============================================================================

class FunctionGemmaDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset using target candidate tools."""

    def __init__(
        self,
        messages_list: List[Any],
        global_tools: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int = 1536,
    ):
        self.messages_list = messages_list
        self.tools_by_name = {t["function"]["name"]: t for t in global_tools}
        self.global_tools = global_tools
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
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
        distractors = [t for n, t in self.tools_by_name.items() if n not in called]
        sample_tools.extend(distractors[:2])

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


# ==============================================================================
# [2/4] Zero-Copy Chunked Cross-Entropy Loss
# ==============================================================================

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


class CleanCausalTrainer(Trainer):
    """Custom causal trainer integrating chunked loss computation."""

    def _wrap_model(self, model, training=True, dataloader=None):
        return model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = chunked_cross_entropy(outputs.logits, labels, chunk_size=256, ignore_index=-100)
        return (loss, outputs) if return_outputs else loss


class LiveProgressCallback(TrainerCallback):
    """Real-time progress reporting callback."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = None
        self.substep = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "=" * 75)
        print(f"TRAINING ACTIVATED: Target Steps: {self.total_steps:,}")
        print(f"Batch Size: {args.per_device_train_batch_size} | Grad Accum: {args.gradient_accumulation_steps}")
        print("=" * 75 + "\n", flush=True)

    def on_substep_end(self, args, state, control, **kwargs):
        self.substep += 1
        if state.global_step == 0:
            vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            print(
                f"  Warm-up batch {self.substep}/{args.gradient_accumulation_steps} completed | "
                f"{time.time() - self.start_time:.1f}s | VRAM: {vram_gb:.2f} GB",
                flush=True,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            elapsed = time.time() - self.start_time if self.start_time else 1.0
            sec_per_step = elapsed / max(step, 1)
            eta_sec = (self.total_steps - step) * sec_per_step
            vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            print(
                f"[Step {step:4d}/{self.total_steps}] "
                f"Loss: {loss:.4f} | LR: {lr:.2e} | "
                f"{sec_per_step:.1f}s/step (ETA: {int(eta_sec//3600)}h {int(eta_sec%3600//60):02d}m) | "
                f"VRAM: {vram_gb:.2f} GB / 14.56 GB",
                flush=True,
            )


# ==============================================================================
# [3/4] Dataset Resolver
# ==============================================================================

def resolve_dataset(data_path: Optional[str], hf_token: Optional[str]) -> str:
    """Finds local dataset file or downloads from Hugging Face Hub."""
    if data_path and os.path.exists(data_path):
        return data_path

    for candidate in ["train.parquet", "../train.parquet"]:
        if os.path.exists(candidate):
            return candidate

    print("Downloading 'train.parquet' from Hugging Face Hub...")
    return hf_hub_download(
        repo_id="thantzinphyo/Hey-Function-Burmese",
        filename="train.parquet",
        repo_type="dataset",
        token=hf_token,
        local_dir=".",
    )


# ==============================================================================
# [4/4] Main Execution
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma on Gemma-Actions-MM")
    parser.add_argument("--model_id", type=str, default="google/functiongemma-270m-it")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./functiongemma-270m-mm")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", None))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # 1. Dataset Loading
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

    print(f"[Dataset] Train: {len(train_msgs):,} | Eval: {len(eval_msgs):,} | Tools: {len(global_tools)}")

    # 2. Tokenizer & Dataset
    print(f"[Tokenizer] Loading '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    train_dataset = FunctionGemmaDataset(
        train_msgs,
        global_tools,
        tokenizer,
        max_seq_length=args.max_seq_length,
    )

    # 3. Model Loading
    print(f"[Model] Loading '{args.model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4. Trainer Configuration
    total_steps = math.ceil(len(train_msgs) / (args.batch_size * args.grad_accum)) * args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=36,
        logging_steps=1,
        disable_tqdm=True,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
    )
    training_args._n_gpu = 1

    trainer = CleanCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=causal_collate_fn,
        callbacks=[LiveProgressCallback(total_steps=total_steps)],
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Execute Training
    trainer.train()

    # 6. Save Model & Tokenizer
    print(f"\n[Saving] Saving fine-tuned artifacts to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SUCCESS: Model and tokenizer saved cleanly.")


if __name__ == "__main__":
    main()
