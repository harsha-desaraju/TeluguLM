


import re
import random
from pathlib import Path
from unicodedata import normalize

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


def preprocess(raw_text):
    """
    The following is the preprocessing that is applied on the text.
    This preprocessing is very specific to the data that it is being applied on.
    """

    other_langs = re.compile(r"[^A-Za-z\u0C00-\u0C7F0-9\s.,!?;:'\"()\[\]{}\-\–—_+=/@#₹%&*<>|\\~`]")
    repeating = re.compile(r'([.*\n\t])\1{3,}')

    p_text = raw_text.strip()
    if '<TEXT>' in p_text:
        p_text = re.findall(r"<TEXT>(.*?)</TEXT>", p_text, re.DOTALL)[0]

    # Remove all text that is not in Telugu or English
    p_text = other_langs.sub("", p_text)

    # Remove all the repeating characters
    p_text = repeating.sub(r'\1', p_text)

    p_text = re.sub(r'</>', '', p_text)

    # Remove wikipedia references
    p_text = re.sub(r'\[\d+\]', '', p_text)

    # Remove data and time stamps
    p_text = re.sub(r'(\d{2}-){2}\d{4} (\d{2}:){2}\d{2}','', p_text)

    # Normalize the text
    p_text = normalize("NFKC", p_text)

    return p_text



if __name__ == '__main__':

    # --------------------- Part-1 ---------------------

    # Dataset creation from data sources that were privately collected.
    test_split = 0.1
    FOLDER_PATH = "/Users/xai/Personal/Projects/Datasets/text_dump/"

    files = list(Path(FOLDER_PATH).rglob("*.txt"))

    texts = []

    for file in files:
        print(file.name)
        with open(file, 'r') as f:
            text = f.read()

        txt_lst = text.split('\n\n-----\n\n')

        p_txt_lst = [preprocess(txt) for txt in txt_lst]

        texts += p_txt_lst

    random.shuffle(texts)

    num_samples = int(test_split*len(texts))
    test_texts = texts[:num_samples]
    train_texts = texts[num_samples:]

    train_df = pd.DataFrame({"text": train_texts})
    test_df = pd.DataFrame({"text": test_texts})


    dataset1 = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })
    print(dataset1)



    # --------------------- Part-2 ---------------------

    # Dataset creation from data sources downloaded from hugging face
    SANGRAHA_DATASET_PATH = "/Users/xai/Personal/Projects/Datasets/ai4bharat-sangraha/"

    data_files = list(Path(SANGRAHA_DATASET_PATH).rglob("*.parquet"))
    data_files = list(map(str, data_files))

    ds = load_dataset("parquet", data_files=data_files)

    web_ds = ds['train'].filter(
        lambda x: x["type"] == "web",
        num_proc=6
    )

    web_ds = web_ds.map(
        lambda x: {"text": preprocess(x['text'])},
        num_proc=10
    )

    web_ds = web_ds.remove_columns([col for col in web_ds.column_names if col != "text"])
    web_ds = web_ds.cast(dataset1['train'].features)
    print(web_ds)

    # --------------------- Part-3 ---------------------
    # combine the datasets

    new_train = concatenate_datasets([dataset1['train'], web_ds])

    dataset1['train'] = new_train

    print(dataset1)

    dataset1.save_to_disk("dataset")











