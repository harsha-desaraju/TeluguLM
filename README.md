
# Telugu-Llama 

Telugu-Llama is a decoder-only Transformer language model trained from scratch for Telugu text modeling. The model is designed to capture linguistic patterns across literary, cultural, and modern Telugu corpora.

**Hugging Face Model Repository:**
https://huggingface.co/harsha-desaraju/telugu-llama-small

---

## Overview

This repository contains a 64M parameter causal language model based on a LLaMA-style architecture. The model is trained on approximately 579 million Telugu tokens and is intended for research and downstream fine-tuning tasks.

---

## Model Architecture

* Architecture: LLaMA-style Transformer (decoder-only)
* Number of parameters: ~64M
* Number of layers: 12
* Hidden size: 512
* Intermediate size: 2048
* Attention heads: 8
* Key-value heads: 4 (Grouped Query Attention)
* Maximum context length: 2048
* Activation function: SiLU
* Positional encoding: RoPE

---

## Tokenizer

* Type: Byte Pair Encoding (BPE)
* Vocabulary size: 16,384
* Token fertility: 1.9
* Training data: Same distribution as model training corpus

The tokenizer is optimized for Telugu script and morphology.

---

## Training Data

The model is trained on a curated Telugu corpus consisting of:

* AI4Bharat Sangraha dataset (`verified/tel`)
* Telugu poetry (classical and modern)
* Ancient music literature
* Telugu film lyrics
* General web text

The dataset provides a mix of formal, literary, and informal Telugu usage.

---

## Training Details

* Number of epochs: 2
* Total tokens: 579 million
* Per-device batch size: 16
* Gradient accumulation steps: 4
* Effective batch size: 64
* Learning rate: 1e-4
* Weight decay: 0.1
* Scheduler: Cosine decay
* Warmup ratio: 0.05
* Max gradient norm: 1.0
* Precision: FP16

**Final evaluation loss:** 3.033

---

## Usage

### Installation

```bash
pip install transformers torch
```

### Loading the model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "harsha-desaraju/telugu-llama-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### Text generation

```python
prompt = "ఎందరో మహానుభావులు అందరికి "
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.95
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Intended Use

The model can be used for:

* Telugu text generation
* Language modeling research
* Pretraining base for fine-tuning (instruction tuning, domain adaptation)

---

## Future Work

* Instruction tuning and alignment
* Benchmark evaluation on Telugu NLP tasks
* Scaling to larger model sizes
* Dataset expansion and improved balancing
