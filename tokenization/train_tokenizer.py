
import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFKC
from preprocess_data import preprocess
from pathlib import Path
from typing import Union

# os.environ["TOKENIZERS_PARALLELISM"] = "true"


def train_tokenizer(
    files: Union[str, list[str]],
    vocab_size= 32768,
    min_frequency=5,
    output_dir="telugu_tokenizer"
):
    """
    Args:
        files: Files to train the tokenizer on
        vocab_size: Vocabulary size
        min_frequency: Minimum token frequency
        output_dir: Directory to save tokenizer
    """

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            byte_fallback=False,

        )
    )

    tokenizer.normalizer = Sequence([NFKC()])

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<cls>",
            "<sep>",
            "<mask>"
        ],
        initial_alphabet=[],  # byte_fallback handles coverage
        show_progress=True
    )

    tokenizer.train(files, trainer=trainer)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save(f"{output_dir}/tokenizer.json")



def data_iterator(files: list[Path]):
    for file in files:
        print(f"Started processing {file.name}")
        ext = file.name.split('.')[-1]
        st = time.time()
        if ext == 'txt':
            with open(file, 'r', encoding="utf-8") as f:
                text = f.read()
            processed_text = preprocess(text)
        elif ext == 'parquet':
            df = pd.read_parquet(file)
            df['text'] = df['text'].apply(preprocess)
            processed_text = df['text'].tolist()
        else:
            print(f"Got different file type {ext}")
        processed_text = '\n\n'.join(processed_text)
        et = time.time()
        print(f"Finished processing {file.name} in {round(et-st, 2)}s\n")
        yield processed_text


if __name__ == '__main__':
    import time

    st = time.time()
    train_tokenizer(files=['tokenization/tokenizer_dataset_1.txt', 'tokenization/tokenizer_dataset_3.txt'], vocab_size=32768)
    et = time.time()
    print(f"Time taken for tokenizer to train: {round(et-st, 2)}s")



