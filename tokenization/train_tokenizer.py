
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFKC

def train_tokenizer(
    # files: Union[str, list[str]],
    # stream: bool=True,
    text_iterator,
    vocab_size=32000,
    min_frequency=2,
    output_dir="telugu_tokenizer"
):
    """
    Args:
        # files: Path to the text file(s)
        # stream: If True, the data is streamed to the tokenizer for training
        text_iterator: The iterator for text
        vocab_size: Vocabulary size
        min_frequency: Minimum token frequency
        output_dir: Directory to save tokenizer
    """

    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]",
            byte_fallback=True
        )
    )

    tokenizer.normalizer = Sequence([NFKC()])

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ],
        initial_alphabet=[]  # byte_fallback handles coverage
    )

    tokenizer.train_from_iterator(
        iterator=text_iterator,
        trainer=trainer
    )

    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save(f"{output_dir}/tokenizer.json")



def data_iterator(files: list[str]):
    for file in files:
        print(file)
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
        yield text


if __name__ == '__main__':

    from pathlib import Path

    file_paths = list(Path('/Users/xai/Personal/Projects/Datasets/text_dump').rglob('*.txt'))
    iterator = data_iterator(file_paths[:2])

    train_tokenizer(iterator, vocab_size=24000)




