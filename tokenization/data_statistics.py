
from tokenizers import Tokenizer
from datasets import load_from_disk
import numpy as np



if __name__ == '__main__':

    tokenizer = Tokenizer.from_file("telugu_tokenizer/telugu_unigram_tokenizer.json")

    dataset = load_from_disk("../dataset_creation/dataset")

    # Calculate the number of tokens
    # 1) Train dataset
    train_dataset = dataset['train']

    train_dataset = train_dataset.map(
        lambda x: {"num_tokens": len(tokenizer.encode(x['text']).tokens)},
        num_proc=10
    )

    # 2) Test dataset
    test_dataset = dataset['test']

    test_dataset = test_dataset.map(
        lambda x: {"num_tokens": len(tokenizer.encode(x['text']).tokens)},
        num_proc=10
    )

    train_tokens = np.sum(train_dataset['num_tokens'])
    print(f"Total number of tokens in training dataset: {train_tokens}")

    test_tokens = np.sum(test_dataset['num_tokens'])
    print(f"Total number of tokens in testing dataset: {test_tokens}")



