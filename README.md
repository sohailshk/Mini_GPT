# 🧠 MiniGPT in JAX & Flax

A clean and minimal implementation of a GPT-style transformer model using [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax), trained on the Tiny Shakespeare dataset.

This project aims to serve as a reference and educational resource for developers and researchers interested in understanding how transformer-based LLMs are structured, initialized, and trained from scratch using modern ML tools.

> ✅ Built with clarity, reproducibility, and learning in mind.

---

## 📚 What You'll Learn

- The structure and working of a simplified GPT architecture.
- How to tokenize and preprocess text data.
- How to define transformer layers using Flax Modules.
- How to train LLMs using JAX’s functional programming model.
- How to evaluate and generate text from a trained model.

---

## 📦 Features

- ✅ Fully JAX/Flax-based Transformer model
- 📜 Trained on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- 🪄 Character-level tokenizer
- 📉 Loss tracking & training visualizations
- 📈 Modular and extensible design
- 🧪 Test-time inference & text generation

---

## 🧪 Training the Model

The notebook provides:
- Epoch-wise average loss
- Loss vs Epoch plot
- Sample text generations after training

---

## 📊 Results

Here’s a sample output after just a few epochs of training on Tiny Shakespeare:

Whate pe gon so hast of th Wome sooked poun go thin And the thass she whall:

> The generated text gets more coherent with longer training and larger model sizes.

## 🤝 Contributions Welcome

This project is designed with extensibility in mind. Future contributions may include:
- [ ] Adding causal masking
- [ ] Supporting multi-GPU (via `pmap`)
- [ ] Switching to sentence-level tokenization (using Hugging Face's `tokenizers`)
- [ ] Integration with TPU via JAX's `jax.experimental.pjit`

