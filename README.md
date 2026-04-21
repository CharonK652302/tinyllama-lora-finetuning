# 🦙 TinyLlama LoRA Fine-tuning + MLflow Tracking

Fine-tuned TinyLlama-1.1B using LoRA on two datasets with MLflow experiment tracking:

1. General instruction following (Alpaca — 2,000 examples)
2. Custom AI/ML domain Q&A dataset (20 examples)

Both training runs tracked with MLflow — hyperparameters, loss curves, BLEU scores.

---

## 🤗 Published Models

| Model | HuggingFace Link |
|---|---|
| General (Alpaca) | [tinyllama-lora-alpaca](https://huggingface.co/CharanGoud652/tinyllama-lora-alpaca) |
| Domain (AI/ML) | [tinyllama-lora-aiml-domain](https://huggingface.co/CharanGoud652/tinyllama-lora-aiml-domain) |

---

## 📊 Training Details

| Config | Value |
|---|---|
| Base model | TinyLlama-1.1B (4-bit quantized via Unsloth) |
| Method | LoRA (rank=16, alpha=32, dropout=0.05) |
| Trainable params | 12,615,680 (1.13% of total) |
| Round 1 dataset | Alpaca-cleaned (2,000 examples) |
| Round 1 steps | 100 steps, final loss: 1.93 |
| Round 2 dataset | Custom AI/ML Q&A (20 examples) |
| Round 2 steps | 60 steps, final loss: 3.02 |
| Evaluation | BLEU scoring (avg: 0.011) |
| GPU | Tesla T4 (Google Colab) |

---

## 📈 MLflow Experiment Tracking

Both training rounds are tracked with MLflow:

```python
# Round 1 logged params
mlflow.log_params({
    "dataset": "yahma/alpaca-cleaned",
    "num_examples": 2000,
    "max_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "gpu": "Tesla T4"
})
mlflow.log_metrics({"final_loss": 1.93, "trainable_params": 12600000})

# Round 2 logged params
mlflow.log_params({
    "dataset": "custom-aiml-qa",
    "num_examples": 20,
    "max_steps": 60,
})
mlflow.log_metrics({"final_loss": 3.02, "avg_bleu": 0.011})
```

Experiment name: `tinyllama-lora-finetuning`

---

## 🧠 What is LoRA?

LoRA (Low-Rank Adaptation) freezes pre-trained model weights and injects trainable rank decomposition matrices into each transformer layer. This reduces trainable parameters by 99% while achieving comparable performance to full fine-tuning.

---

## 🚀 Run in Google Colab

Open `tinyllama_lora_finetuning.ipynb` in Google Colab with T4 GPU runtime. Run all cells sequentially.

The notebook includes:
- Model loading with 4-bit quantization (Unsloth)
- LoRA config + training (both rounds)
- BLEU evaluation
- MLflow experiment tracking cells
- HuggingFace Hub push

---

## 📂 Custom Dataset

`dataset/aiml_dataset.json` contains 20 AI/ML domain Q&A pairs covering RAG, LangChain, FAISS, LoRA, transformers, embeddings, and more.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning | LoRA + PEFT |
| Quantization | 4-bit (Unsloth) |
| Experiment Tracking | MLflow |
| Evaluation | BLEU (sacrebleu) |
| Training Platform | Google Colab (Tesla T4) |
| Model Hub | HuggingFace Hub |
| Framework | PyTorch + Transformers |

---

## 📈 Evaluation

BLEU scoring on 5 ML concept questions:

- Average BLEU: 0.011
- Note: Low BLEU is expected for open-ended generation with small models. Coherent answers observed on domain-specific questions.

---

## 📂 Project Structure

```
tinyllama-lora-finetuning/
├── dataset/
│   └── aiml_dataset.json          # Custom AI/ML Q&A dataset
├── tinyllama_lora_finetuning.ipynb # Training notebook + MLflow cells
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

**Sai Charan Goud K**
AI/ML Engineer | LoRA · LLM Fine-tuning · HuggingFace · MLflow
[GitHub](https://github.com/CharonK652302) ·
[LinkedIn](https://www.linkedin.com/in/sai-charan-goud-kowlampet-007654284/) ·
[HuggingFace](https://huggingface.co/CharanGoud652)
