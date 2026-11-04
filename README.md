# Hey-Function-Burmese & Hey-270M-Alpha

### Burmese & English Function-Calling Dataset and On-Device Agent Suite

Official open-source repository for training, fine-tuning, and evaluating lightweight on-device Language Models on the **Hey-Function-Burmese** dataset. This repository accompanies the **Hey-270M-Alpha** function-calling model.

---

## Resources

- **Fine-Tuned Model:** [thantzinphyo/Hey-270M-Alpha](https://huggingface.co/thantzinphyo/Hey-270M-Alpha)
- **Dataset on Hugging Face:** [thantzinphyo/Hey-Function-Burmese](https://huggingface.co/datasets/thantzinphyo/Hey-Function-Burmese)
- **Base Architecture:** `google/functiongemma-270m-it` (270 Million parameters)

---

## Benchmark Evaluation Results

Evaluated on a held-out test split of 200 samples across linguistic registers:

| Register / Language | Samples | Fn Acc (%) |
| :--- | :---: | :---: |
| English | 83 | 75.9 |
| Burmese | 38 | 60.5 |
| Code-Switch | 79 | 45.6 |

*Fn Acc (Tool Selection Accuracy): Measures the percentage of instances where the model correctly identified and triggered the exact target tool.*  
*Note: Training was conducted as an Initial Baseline (1 Epoch) on a Kaggle NVIDIA Tesla T4 GPU with custom chunked cross-entropy.*

---

## Dataset Overview

- **Total Examples:** 42,673 (38,833 Train / 3,840 Eval)
- **Tool Taxonomy:** 61 distinct tools across device controls, communications, navigation, utilities, smart home, and vehicle IoT.
- **Language Composition:** English (15,898), Burmese (13,096), and English-Burmese Code-Switching (13,679).

---

## Installation

```bash
git clone https://github.com/thantzinphyo-gg/Gemma-Actions-MM.git
cd Gemma-Actions-MM
pip install -r requirements.txt
```

If accessing gated or private model weights, export your Hugging Face access token:

```bash
export HF_TOKEN="your_huggingface_token"
```

---

## Evaluation

Evaluate the pre-trained `Hey-270M-Alpha` model directly from Hugging Face Hub (the dataset will download automatically if not present locally):

```bash
python evaluate.py \
    --model_path thantzinphyo/Hey-270M-Alpha \
    --samples 200
```

---

## Training

The training script reproduces the fine-tuning process using the `CleanCausalTrainer` architecture:
- Target function candidate extraction
- Dynamic batch padding with causal collation
- Zero-copy chunked cross-entropy (prevents VRAM allocation spikes on large vocabularies)
- FP16 mixed-precision training with gradient checkpointing
- Real-time step progress callback

Run fine-tuning:

```bash
python train.py \
    --model_id google/functiongemma-270m-it \
    --output_dir ./functiongemma-270m-mm \
    --batch_size 1 \
    --grad_accum 32 \
    --learning_rate 2e-5 \
    --epochs 1
```

---

## Citation

If you use this model, dataset, or training pipeline in your research or applications, please cite:

```bibtex
@misc{hey_270m_alpha_2026,
  title={Hey-270M-Alpha: A Lightweight On-Device Burmese & English Function-Calling Agent},
  author={Thant Zin Phyo},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/thantzinphyo/Hey-270M-Alpha}}
}

@dataset{hey_function_burmese_2026,
  title={Hey-Function-Burmese: A Burmese & English Function-Calling Dataset for Lightweight / Edge LLMs},
  author={Thant Zin Phyo},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/thantzinphyo/Hey-Function-Burmese}}
}
```

---

## License

- Code and Model: [Apache 2.0 License](LICENSE), subject to the upstream [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- Dataset: [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
