# TinyLlama LoRA Fine-tuning

Fine-tuned TinyLlama-1.1B using LoRA on two datasets:
1. General instruction following (Alpaca)
2. Custom AI/ML domain Q&A dataset

## 🤗 Published Models

| Model | HuggingFace Link |
|---|---|
| General (Alpaca) | [tinyllama-lora-alpaca](https://huggingface.co/CharanGoud652/tinyllama-lora-alpaca) |
| Domain (AI/ML) | [tinyllama-lora-aiml-domain](https://huggingface.co/CharanGoud652/tinyllama-lora-aiml-domain) |

## 📊 Training Details

| Config | Value |
|---|---|
| Base model | TinyLlama-1.1B (4-bit quantized) |
| Method | LoRA (rank=16, alpha=32) |
| Trainable params | 12,615,680 (1.13% of total) |
| Round 1 dataset | Alpaca-cleaned (2,000 examples) |
| Round 1 steps | 100 steps, final loss: 1.9262 |
| Round 2 dataset | Custom AI/ML Q&A (20 examples) |
| Round 2 steps | 60 steps, final loss: 3.0216 |
| Evaluation | BLEU scoring (avg: 0.0110) |
| GPU | Tesla T4 (Google Colab) |

## 🧠 What is LoRA?

LoRA (Low-Rank Adaptation) freezes pre-trained model weights
and injects trainable rank decomposition matrices into each
transformer layer. This reduces trainable parameters by 99%
while achieving comparable performance to full fine-tuning.

## 🚀 Run in Google Colab

Open `tinyllama_lora_finetuning.ipynb` in Google Colab
with T4 GPU runtime. Run all cells sequentially.

## 📂 Custom Dataset

`dataset/aiml_dataset.json` contains 20 AI/ML domain
Q&A pairs covering RAG, LangChain, FAISS, LoRA,
transformers, embeddings, and more.

## 📈 Evaluation

BLEU scoring on 5 ML concept questions:
- Average BLEU: 0.0110
- Note: Low BLEU is expected for open-ended generation
  with small models. Coherent answers observed on
  domain-specific questions.

## 👨‍💻 Author

**Sai Charan Goud K**
AI/ML Engineer | LoRA · LLM Fine-tuning · HuggingFace
[GitHub](https://github.com/CharonK652302) ·
[LinkedIn](https://www.linkedin.com/in/sai-charan-goud-kowlampet-007654284/)
