# Hey-Function-Burmese

> Lightweight On-Device Burmese (Myanmar) & English Function-Calling Suite

[Model (Hey-270M-Alpha)](https://huggingface.co/thantzinphyo/Hey-270M-Alpha) • [Dataset (Hey-Function-Burmese)](https://huggingface.co/datasets/thantzinphyo/Hey-Function-Burmese) • [Base Model (FunctionGemma-270M)](https://huggingface.co/google/functiongemma-270m-it)

Official open-source repository for training, fine-tuning, and evaluating lightweight on-device function-calling agents for Burmese, English, and code-switching natural language instructions.

---

### Benchmark Results (Held-out Test Set)

| Register / Language | Samples | Fn Acc (%) |
| :--- | :---: | :---: |
| English | 83 | 75.9 |
| Burmese | 38 | 60.5 |
| Code-Switch | 79 | 45.6 |

*Fn Acc (Tool Selection Accuracy) on 200 held-out samples. Initial Baseline (1 Epoch).*

---

### Dataset Summary

- **Total Samples:** 42,673 (38,833 Train / 3,840 Eval) across 61 mobile & edge tools.
- **Language Composition:** English (15,898), Burmese (13,096), and Code-Switching (13,679).

---

### Installation

```bash
git clone https://github.com/thantzinphyo-gg/Hey-Function-Burmese.git
cd Hey-Function-Burmese
pip install -r requirements.txt
```

If accessing gated model weights, set your Hugging Face access token:

```bash
export HF_TOKEN="your_huggingface_token"
```

---

### Evaluation

Evaluate the pre-trained `Hey-270M-Alpha` model directly (dataset downloads automatically from Hugging Face if not found locally):

```bash
python evaluate.py --model_path thantzinphyo/Hey-270M-Alpha --samples 200
```

---

### Training Pipeline

Fine-tune FunctionGemma using the memory-efficient `CleanCausalTrainer` architecture with chunked cross-entropy and response masking:

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

### Citation

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

### License

- Code & Model: [Apache 2.0 License](LICENSE), subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- Dataset: [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
