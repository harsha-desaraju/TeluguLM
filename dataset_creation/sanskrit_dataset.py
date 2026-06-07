"""
Transliterate Sanskrit Devanagari dataset to Telugu script
Push the dataset to Hugging face
"""

from aksharamukha import transliterate
from joblib import Parallel, delayed
from datasets import Dataset, Features, Value



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

    print(len(san_text_batches))

    with Parallel(n_jobs=NUM_PROC) as parallel:
        tel_text_batches = parallel([delayed(transliterate_akshara)(batch) for batch in san_text_batches])


    # Convert to hugging face dataset
    tel_text_batches = [{"text": batch} for batch in tel_text_batches]

    ds_features = Features({
        "text": Value("string")
    })

    ds = Dataset.from_list(tel_text_batches, features=ds_features)
    print(ds)

    ds.push_to_hub(
        "harsha-desaraju/telugu-script-sanskrit-text",
        commit_message="Upload the dataset"
    )

