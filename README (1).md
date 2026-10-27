# Tiny Transformer: A GPT-Style Language Model Built from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/116ex_GGKQjWMR7Hrb-E_GomRCpQzIPMa?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

A small decoder-only Transformer (about 10M parameters) written in PyTorch and trained from scratch to generate short children's stories. The project covers the full language-modelling pipeline: tokenizer training, data preparation, model architecture, training, evaluation, text generation, and a small interactive demo.

Built as a lab project for **PUSL 3205** (BSc Data Science, University of Plymouth / NSBM Green University).

## Highlights

- **GPT-style decoder-only Transformer** implemented from scratch: masked (causal) multi-head self-attention, token and positional embeddings, feed-forward layers with GELU, residual connections, and pre-LayerNorm.
- **Custom BPE tokenizer** with a 4,096-token vocabulary, trained on the dataset with Hugging Face `tokenizers`.
- **Efficient training:** mixed precision (fp16), fused attention (`scaled_dot_product_attention`), AdamW, OneCycle learning-rate schedule, and gradient clipping.
- **Evaluation** using validation loss and perplexity, compared against a character-level baseline.
- **Controlled experiments** across 7 configurations varying depth, number of attention heads, and training length.
- **Sampling controls:** temperature, top-k, and top-p (nucleus) sampling.
- **Gradio demo** for generating stories from your own prompts.

## Dataset

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (`roneneldan/TinyStories`): short, simple stories written with a limited vocabulary. This project uses the first **50,000 training stories** and **2,000 validation stories**.

## Model

| Setting | Value |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Context length | 256 tokens |
| Vocabulary | 4,096 (BPE) |
| Dropout | 0.1 |
| Parameters | ~10M |

## Results

> Fill in these numbers after running the notebook, then delete this note.

| Metric | Value |
|---|---|
| Final validation loss (per token) | _X.XX_ |
| Validation perplexity (per token) | _X.XX_ |
| Character-level baseline (loss / perplexity per char) | 1.225 / 3.40 |
| Training time (T4 GPU) | _~XX min_ |

Add the training-loss plot here:

```
![Training and validation loss](images/loss_curve.png)
```

### Sample output

Prompt: `Once upon a time`

```
Paste a short generated story here.
```

### Experiment summary

The notebook compares smaller models (embedding size 256) to see how each setting affects validation loss.

| Layers | Heads | Epochs | Parameters | Val loss | Perplexity |
|---|---|---|---|---|---|
| 2 | 4 | 6 | | | |
| 4 | 4 | 6 | | | |
| 6 | 4 | 6 | | | |
| 4 | 2 | 6 | | | |
| 4 | 8 | 6 | | | |
| 4 | 4 | 3 | | | |
| 4 | 4 | 12 | | | |

Key takeaways: _add 2 to 3 sentences on what you found, for example the effect of depth versus training length._

## Getting Started

### Run in Colab (recommended)

1. Click the **Open in Colab** badge at the top.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. Full training takes roughly 15 to 25 minutes, and the optional experiments add about 30 to 40 minutes.

### Run locally

```bash
git clone https://github.com/MenadiDev/tiny-transformer-tinystories.git
cd tiny-transformer-tinystories
pip install torch datasets tokenizers gradio matplotlib jupyter
jupyter notebook tiny_transformer_lab3_v2_bpe.ipynb
```

A CUDA GPU is strongly recommended. On CPU, reduce the number of training steps.

## Notebook Structure

1. Load and explore the dataset
2. Preprocess: train the BPE tokenizer and encode the corpus
3. Prepare next-token prediction batches
4. Build the Transformer architecture
5. Initialise the model, loss, and optimizer
6. Train and monitor loss
7. Evaluate: validation loss and perplexity
8. Generate stories and compare sampling settings
9. Run model-setting experiments
10. Save and reload the model
11. Optional Gradio chat window

## What I Learned

- How attention, causal masking, and positional embeddings fit together in a decoder-only Transformer.
- Why subword tokenization helps a small model focus on grammar and story structure instead of spelling.
- How learning-rate scheduling, mixed precision, and gradient clipping affect training speed and stability.
- How temperature, top-k, and top-p trade diversity against coherence in generated text.

## Limitations and Next Steps

- The model is small and trained on a narrow dataset, so it only writes simple children's stories and can lose coherence over longer text.
- Possible next steps: train on the full dataset, add weight tying, try a larger context window, and compare against a pretrained model fine-tuned on the same data.

## Tech Stack

Python, PyTorch, Hugging Face Datasets and Tokenizers, Matplotlib, Gradio, Google Colab

## Acknowledgements

- [TinyStories](https://arxiv.org/abs/2305.07759) dataset by Eldan and Li.
- Inspired by Andrej Karpathy's *nanoGPT* and "Let's build GPT" tutorial.

## Author

**Dasithma Menadi**: [LinkedIn](https://www.linkedin.com/in/dasithma-menadi-8565032a8) | [GitHub](https://github.com/MenadiDev)
