
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="telugu_tokenizer/telugu_unigram_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)


dataset = load_from_disk("../dataset_creation/dataset")
dataset = dataset['test']


fertility = []

for txt in tqdm(dataset['text']):
    n_words = len(txt.split())
    n_tokens = len(tokenizer.encode(txt))

    if n_words != 0 and n_tokens != 0:
        fertility.append(n_tokens/n_words)
    else:
        print(txt)

avg_fertility = sum(fertility)/len(fertility)

print(f"The average fertility of the tokenizer is: {round(avg_fertility, 2)}")



# print the tokenization for some sentences
texts = [
    "ప్రభుత్వం ప్రకటించిన కొత్త విధానాల ప్రకారం రైతులకు ఆర్థిక సహాయం అందించబడుతుంది.",
    "అబ్బా!!! ఇది ఏమిటి???",
    "హాహాహా ఇది చాలా ఫన్నీగా ఉంది.",
    "ఏమైంది... ఎందుకు ఆలస్యం?",
    "నేను meeting కి వెళ్తున్నాను.",
    "ఇది చాలా interesting విషయం.",
    "అతను project complete చేశాడు.",
    "మీరు report submit చేశారా?"
]

for text in texts:
    tokens = tokenizer.encode(text)
    # text_tokens = [tokenizer.decode(token) for token in tokens]
    text_tokens = tokenizer.decode(tokens)
    print(text_tokens)




