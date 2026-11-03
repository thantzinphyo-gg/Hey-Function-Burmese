# Gemma-Actions-MM

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/thantzinphyo/Gemma-Actions-MM)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Model Architecture](https://img.shields.io/badge/Model-FunctionGemma--270M-orange)](https://huggingface.co/google/functiongemma-270m-it)

Official repository for fine-tuning and evaluating lightweight on-device Language Models on the **Gemma-Actions-MM** dataset — a multilingual (English, Burmese, and Code-Switching) function-calling benchmark for edge AI.

---

## Overview

- **Dataset on Hugging Face:** [thantzinphyo/Gemma-Actions-MM](https://huggingface.co/datasets/thantzinphyo/Gemma-Actions-MM)
- **Total Examples:** 42,673 (38,833 Train / 3,840 Eval)
- **Tool Taxonomy:** 61 distinct tools across device controls, communications, navigation, utilities, smart home, and vehicle IoT.
- **Languages:** English (15,898), Burmese (13,096), and English-Burmese Code-Switching (13,679).

---

## Installation

```bash
git clone https://github.com/thantzinphyo-gg/Gemma-Actions-MM.git
cd Gemma-Actions-MM
pip install -r requirements.txt
```

Set your Hugging Face access token (optional, only required if accessing gated model weights):

```bash
export HF_TOKEN="your_huggingface_token"
```

---

## Training

The training script implements:
- FP16 mixed-precision training.
- Chunked cross-entropy to prevent high VRAM allocations on 256K vocabulary models.
- Dynamic collation with zero-redundancy tool padding.

Run fine-tuning on a single GPU (e.g., Nvidia T4 / RTX 3090):

```bash
python train.py \
    --model_id google/functiongemma-270m-it \
    --output_dir ./functiongemma-270m-mm \
    --batch_size 2 \
    --grad_accum 16 \
    --learning_rate 2e-5 \
    --max_seq_length 1536 \
    --epochs 1
```

---

## Evaluation

Evaluate the fine-tuned model on held-out samples:

```bash
python evaluate.py \
    --model_path ./functiongemma-270m-mm \
    --data_path train.parquet \
    --samples 200
```

This outputs tool selection accuracy, argument match accuracy, and overall exact match percentages across English, Burmese, and Code-Switching queries.

---

## Citation

If you use this dataset or training pipeline in your research, please cite:

```bibtex
@dataset{gemma_actions_mm_2026,
  title={Gemma-Actions-MM: A Burmese & English Function-Calling Dataset for Lightweight / Edge LLMs},
  author={Thant Zin Phyo},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/thantzinphyo/Gemma-Actions-MM}}
}
```

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](LICENSE) license.
