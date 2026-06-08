"""
Transliterate Sanskrit Devanagari dataset to Telugu script
Push the dataset to Hugging face
"""
import pandas as pd
from aksharamukha import transliterate
from joblib import Parallel, delayed
from datasets import Dataset, Features, Value, concatenate_datasets


def transliterate_akshara(devanagari_text):
    telugu_text = transliterate.process('Devanagari', 'Telugu', devanagari_text)
    return telugu_text



def transliterate_batch(lines_list):
    batch_size = 100
    target = []
    for i in range(0, len(lines_list), batch_size):
        src_lines = lines_list[i: i+batch_size]
        src_lines = "".join(src_lines)
        tar_lines = transliterate_akshara(src_lines)
        target.append(tar_lines)
    return target



if __name__ == '__main__':

    # Source 1: https://www.kaggle.com/datasets/aluminium13/vyakaran
    df1 = pd.read_csv("tokens.csv")
    df1 = df1[['Sanskrit']]

    df1['text'] = df1['Sanskrit'].apply(lambda x: transliterate_akshara(x))
    df1 = df1[['text']]
    df1 = df1.to_dict(orient='records')
    ds1 = Dataset.from_list(df1, features=Features({"text": Value("string")}))
    print(ds1)


    # Source 2: https://www.kaggle.com/datasets/jammikunal/sanskrit-data
    df2 = pd.read_csv("sanskrit_data.csv")[["text"]]
    df2["telugu_text"] = df2['text'].apply(lambda x: transliterate_akshara(x))
    df2 = df2.drop(columns=['text']).rename(columns={'telugu_text': "text"})
    df2 = df2.to_dict(orient="records")

    ds2 = Dataset.from_list(df2, features=Features({"text": Value("string")}))
    print(ds2)


    # Source 3: https://www.kaggle.com/datasets/preetsojitra/sanskrit-text-corpus
    BATCH_SIZE = 100
    NUM_PROC = 8

    dataset_path = 'archive/train.txt'

    with open(dataset_path) as f:
        san_text_lines = f.readlines()


    san_text_batches = []
    for i in range(0, len(san_text_lines), BATCH_SIZE):
        san_text_batch = san_text_lines[i:i+BATCH_SIZE]
        san_text_batch = "".join(san_text_batch)
        san_text_batches.append(san_text_batch)

    with Parallel(n_jobs=NUM_PROC) as parallel:
        tel_text_batches = parallel([delayed(transliterate_akshara)(batch) for batch in san_text_batches])


    # Convert to hugging face dataset
    tel_text_batches = [{"text": batch} for batch in tel_text_batches]

    ds_features = Features({
        "text": Value("string")
    })

    ds3 = Dataset.from_list(tel_text_batches, features=ds_features)
    print(ds3)

    ds = concatenate_datasets([ds1, ds2, ds3])
    print(ds)

    ds.push_to_hub(
        "harsha-desaraju/telugu-script-sanskrit-text",
        commit_message="Upload the dataset"
    )

