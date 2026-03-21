
from tokenizers import Tokenizer, models, trainers, normalizers
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

from datasets import load_from_disk


def data_iterator(ds):
    for example in ds['train']:
        yield example['text']



if __name__ == '__main__':

    dataset = load_from_disk("../dataset_creation/dataset")
    iterator = data_iterator(dataset)

    # 1. Initialize tokenizer with Unigram model
    telugu_tokenizer = Tokenizer(models.Unigram())

    # 2. Normalization (important for Telugu Unicode consistency)
    telugu_tokenizer.normalizer = normalizers.Sequence([NFKC()])

    # 3. Pre-tokenization
    telugu_tokenizer.pre_tokenizer = Metaspace(replacement="_")

    telugu_tokenizer.decoder = MetaspaceDecoder(replacement="_")

    # 4. Trainer
    trainer = trainers.UnigramTrainer(
        vocab_size=16384,  # adjust as needed (16k–32k typical)
        special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
        ],
        unk_token="[UNK]", show_progress=True
    )

    telugu_tokenizer.train_from_iterator(iterator, trainer)

    # 6. Save tokenizer
    telugu_tokenizer.save("telugu_tokenizer/telugu_unigram_tokenizer.json")

    print("Tokenizer trained and saved!")