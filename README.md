# Tiny Transformer Language Model (BPE Tokenizer)

A GPT-style, decoder-only Transformer trained from scratch in PyTorch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The model learns to generate short, simple children's stories one token at a time.

This is **v2** of PUSL 3205 Lab 3. It improves on a character-level baseline with a subword (BPE) tokenizer, more data, a larger model, mixed-precision training, a learning-rate schedule and nucleus (top-p) sampling.

## Highlights

- Custom **Byte-Pair Encoding tokenizer** (4,096-token vocabulary) trained on the corpus with Hugging Face `tokenizers`
- Transformer written from scratch: token and positional embeddings, masked multi-head self-attention, feed-forward blocks, residual connections and pre-LayerNorm
- **12.32M parameters** (6 layers, 6 heads, 384-dim embeddings, 256-token context)
- Fused attention (`F.scaled_dot_product_attention`) and **fp16 mixed precision**
- AdamW with OneCycle LR (warm-up and decay), gradient clipping and weight decay
- Text generation with temperature, top-k and top-p (nucleus) sampling
- Experiments on depth, number of heads and training length
- Interactive **Gradio** demo for story generation

## Dataset

| | |
|---|---|
| Source | `roneneldan/TinyStories` (2,119,719 train / 21,990 validation stories) |
| Subset used | First 50,000 training stories and 2,000 validation stories |
| Average story length | ~850 characters, ~166 words |
| Train / val tokens | 11,356,695 / 399,129 |
| Average characters per token | 3.99 |

An `<eos>` token marks the end of each story. Training uses next-token prediction: for a block of tokens `x = tokens[i : i+T]`, the target is `y = tokens[i+1 : i+T+1]`.

## Model Architecture

| Setting | Value |
|---|---|
| Vocabulary size | 4,096 (BPE) |
| Context length (`block_size`) | 256 |
| Layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Dropout | 0.1 |
| Parameters | 12.32M |

Design details:

- **Pre-LayerNorm** transformer blocks with two residual connections each
- **Weight tying** between the token embedding and output head
- **Scaled initialisation** of residual projections (`0.02 / sqrt(2 * n_layer)`) for stable deeper training
- **GELU** feed-forward layers with a 4x hidden expansion

## Training Setup

| Setting | Value |
|---|---|
| Epochs x steps | 10 x 600 (6,000 steps) |
| Batch size | 64 |
| Peak learning rate | 1e-3 (OneCycleLR, 5% warm-up) |
| Optimizer | AdamW (betas 0.9/0.95, weight decay 0.1) |
| Gradient clipping | 1.0 |
| Precision | fp16 autocast with GradScaler |
| Hardware | Google Colab T4 GPU (~16 min total) |

## Results

Final model on the validation set:

| Metric | Value |
|---|---|
| Validation loss (per token) | 1.8333 |
| Perplexity (per token) | 6.25 |
| Validation loss (per character) | 0.4596 |
| Perplexity (per character) | 1.58 |

Per-character loss is computed as the per-token loss divided by the average characters per token, so it can be compared with the character-level baseline.

Validation loss fell steadily from 2.67 after epoch 1 to 1.84 after epoch 10, with train and validation loss staying close together (no significant overfitting).

### Experiments

Three smaller configurations (`n_embd=256`, 250 steps per epoch) were compared:

| Layers | Heads | Epochs | Params | Val loss | Perplexity |
|:-:|:-:|:-:|--:|--:|--:|
| 2 | 4 | 6 | 2.69M | 2.577 | 13.16 |
| 4 | 2 | 6 | 4.27M | 2.506 | 12.26 |
| 4 | 4 | 3 | 4.27M | 2.906 | 18.29 |

Takeaways: more depth helped, and longer training clearly mattered (3 epochs vs 6). The full-size model, with more data and training steps, performs far better than any of these smaller runs.

### Sample output

Prompt: `Once upon a time`

> Once upon a time, there was a little girl named Lily. She had a favorite jacket that she loved to wear. One day, Lily's friend came over to play. Lily showed her jacket and said, "Look, Mommy! It's my favorite jacket!" ...

Sampling settings change the output noticeably: low temperature (0.5) gives safe, repetitive stories, while high temperature (1.1, top-p 1.0) produces more creative but less coherent text.

## Getting Started

### Requirements

- Python 3.9+
- A CUDA GPU is recommended (the notebook falls back to CPU, but training is slow)

```bash
pip install torch datasets tokenizers gradio matplotlib
```

### Run

1. Open `tiny_transformer_lab3_v2_bpe.ipynb` in Google Colab (set the runtime to GPU) or Jupyter.
2. Run the cells in order. The notebook will:
   1. Download and explore TinyStories
   2. Train the BPE tokenizer and encode the corpus
   3. Build, train and evaluate the Transformer
   4. Generate sample stories and run the model-setting experiments
   5. Save the model and launch the Gradio demo

### Saved artifacts

| File | Contents |
|---|---|
| `tiny_transformer_bpe.pt` | Model weights and config |
| `tokenizer.json` | Trained BPE tokenizer |

### Load the trained model

```python
import torch
from tokenizers import Tokenizer

ck = torch.load("tiny_transformer_bpe.pt", map_location="cpu")
model = TinyTransformer(Config(**ck["config"]))
model.load_state_dict(ck["state_dict"])
model.eval()

tok = BPETokenizer(Tokenizer.from_file("tokenizer.json"))
print(sample_text(model, tok, "Once upon a time", n=300))
```

`TinyTransformer`, `Config`, `BPETokenizer` and `sample_text` are defined in the notebook.

## Gradio Demo

The final cell launches a web interface where you can enter a story prompt and adjust temperature, top-p and maximum length. With `share=True`, Colab provides a temporary public link.

## What Improves Performance

- **BPE tokenization** puts capacity into grammar and story structure rather than spelling, and shortens sequences
- **Causal self-attention and multi-head attention** capture long-range dependencies such as a character's name several sentences earlier
- **Residual connections, pre-LayerNorm and scaled init** keep deeper models stable
- **More data and longer training** was the largest single factor over the baseline
- **AdamW, LR warm-up/decay, gradient clipping** give faster and more stable convergence
- **Mixed precision and fused attention** allow more training steps in the same time
- **Top-k and top-p sampling** balance diversity and coherence at generation time

## Limitations

- Trained on only 50,000 stories, so output is simple and sometimes repetitive or incoherent, especially at higher temperatures
- Small context window (256 tokens), so very long stories lose coherence
- Samples show occasional logical slips (e.g. objects that do not behave consistently)

## Acknowledgements

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li
- Hugging Face `datasets` and `tokenizers`
- Coursework: PUSL 3205, Lab 3
