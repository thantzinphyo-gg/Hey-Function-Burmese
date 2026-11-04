# Hey-Function-Burmese

> Lightweight On-Device Burmese (Myanmar) & English Function-Calling Suite

[Model (Hey-270M-Alpha)](https://huggingface.co/thantzinphyo/Hey-270M-Alpha) • [Dataset (Hey-Function-Burmese)](https://huggingface.co/datasets/thantzinphyo/Hey-Function-Burmese) • [Base Model (FunctionGemma-270M)](https://huggingface.co/google/functiongemma-270m-it)

Official open-source repository for training, fine-tuning, and evaluating lightweight on-device function-calling agents for Burmese, English, and code-switching natural language instructions.

---

### Benchmark Results (Held-out Test Set)

| Register / Language | Samples | Fn Acc (%) | Arg Acc (%) | Exact Match (EM) |
| :--- | :---: | :---: | :---: | :---: |
| **English** | 71 | 69.0% | 67.6% | 67.6% |
| **Burmese** | 47 | 44.7% | 44.7% | 44.7% |
| **Code-Switch** | 73 | 52.1% | 52.1% | 52.1% |
| **Overall** | **191** | **56.5%** | **56.0%** | **56.0%** |

- **Fn Acc (Function Accuracy):** Percentage of samples where all predicted tool names match ground truth.
- **Arg Acc (Argument Accuracy):** Percentage of samples where tool names AND all arguments match ground truth.
- **Exact Match (EM):** Strict end-to-end match.

### Per-Tool Accuracy
*Evaluated at tool invocation level (minimum 3 samples):*

| Tool Name | Correct | Total | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| `turn_off_flashlight` | 12 | 12 | 100.0% |
| `check_traffic` | 3 | 3 | 100.0% |
| `open_website` | 4 | 4 | 100.0% |
| `open_wifi_settings` | 8 | 9 | 88.9% |
| `create_contact` | 30 | 35 | 85.7% |
| `turn_on_flashlight` | 15 | 18 | 83.3% |
| `navigate_to` | 13 | 16 | 81.2% |
| `send_email` | 18 | 26 | 69.2% |
| `show_map` | 17 | 29 | 58.6% |
| `create_calendar_event` | 33 | 61 | 54.1% |
| `set_volume_level` | 2 | 4 | 50.0% |
| `play_music` | 12 | 26 | 46.2% |
| `control_media` | 1 | 6 | 16.7% |

---

### Dataset Summary

- **Total Samples:** 42,673 (38,833 Train / 3,840 Eval) across 61 mobile & edge tools.
- **Language Composition:** English (15,898), Burmese (13,096), and Code-Switching (13,679).

---

### Quick Start with Jupyter Notebook

An end-to-end, all-in-one Jupyter notebook is available under `notebooks/`:

- [`Hey_Function_270M_Training_and_Benchmark.ipynb`](notebooks/Hey_Function_270M_Training_and_Benchmark.ipynb)

This notebook runs out of the box on **Kaggle** (Tesla T4 GPU) or **Google Colab**, containing the complete pipeline: data download, memory-efficient candidate collation, full 1-epoch training with chunked cross-entropy, and comprehensive evaluation.

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
    --max_seq_length 2800 \
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
