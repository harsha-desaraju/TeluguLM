

import torch
from pathlib import Path


if __name__ == '__main__':

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_fol_path = Path("telugu_llama")
    # model_fol_path = Path("telugu-gpt/checkpoint-107300")

    tokenizer = AutoTokenizer.from_pretrained(model_fol_path)
    model = AutoModelForCausalLM.from_pretrained(model_fol_path)

    text = "ఎందరో మహానుభావులు అందరికి "
    # text = "అతను అక్కడికి వెళ్ళాడు కానీ"

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=500,
            do_sample=True,
            temperature=1.0,
            # top_k=20,
            # top_p=0.9
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))