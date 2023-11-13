import os
import re
import time
from pathlib import Path
from typing import List, Union

from tokenizers import Tokenizer, decoders, models, processors, trainers
from tokenizers.processors import TemplateProcessing

# TODO: How to import any_file_fasta_reader from utils
#from utils import any_file_fasta_reader
from transformers import BatchEncoding, PreTrainedTokenizerFast

PathLike = Union[str, Path]


def read_fasta_only_seq(fasta_file: PathLike) -> List[str]:
    """Reads fasta file sequences without description tag."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return lines[1::2]


def any_file_fasta_reader(fasta_file: PathLike) -> List[str]:
    if os.path.isdir(fasta_file):
        sequences_raw = []
        for p in Path(fasta_file).glob("*.fasta"):
            sequences_raw.extend(read_fasta_only_seq(p))
    elif os.path.isfile(fasta_file):
        sequences_raw = read_fasta_only_seq(fasta_file)
    else:
        raise ValueError(
            "Kindly enter a filepath to a directory containing many .fasta files "
            "or a filepath to a single .fasta file"
        )

    return sequences_raw


# Assign a unique character to each codon so that we can use it as an
# input token to a BPE tokenizer. This implements a codon-pair encoding.
CODON_TO_CHAR = {
    "TCG": "A",
    "GCA": "B",
    "CTT": "C",
    "ATT": "D",
    "TTA": "E",
    "GGG": "F",
    "CGT": "G",
    "TAA": "H",
    "AAA": "I",
    "CTC": "J",
    "AGT": "K",
    "CCA": "L",
    "TGT": "M",
    "GCC": "N",
    "GTT": "O",
    "ATA": "P",
    "TAC": "Q",
    "TTT": "R",
    "TGC": "S",
    "CAC": "T",
    "ACG": "U",
    "CCC": "V",
    "ATC": "W",
    "CAT": "X",
    "AGA": "Y",
    "GAG": "Z",
    "GTG": "a",
    "GGT": "b",
    "GCT": "c",
    "TTC": "d",
    "AAC": "e",
    "TAT": "f",
    "GTA": "g",
    "CCG": "h",
    "ACA": "i",
    "CGA": "j",
    "TAG": "k",
    "CTG": "l",
    "GGA": "m",
    "ATG": "n",
    "TCT": "o",
    "CGG": "p",
    "GAT": "q",
    "ACC": "r",
    "GAC": "s",
    "GTC": "t",
    "TGG": "u",
    "CCT": "v",
    "GAA": "w",
    "TCA": "x",
    "CAA": "y",
    "AAT": "z",
    "ACT": "0",
    "GCG": "1",
    "GGC": "2",
    "CTA": "3",
    "AAG": "4",
    "AGG": "5",
    "CAG": "6",
    "AGC": "7",
    "CGC": "8",
    "TTG": "9",
    "TCC": "!",
    "TGA": "@",
    "XXX": "*",
}

CHAR_TO_CODON = {v: k for k, v in CODON_TO_CHAR.items()}


def group_and_contextualize(seq: str, k: int = 3):
    return " ".join(CODON_TO_CHAR.get(seq[i : i + k], "") for i in range(0, len(seq), k))


# TODO:"TypeError: type 'tokenizers.Tokenizer' is not an acceptable base type"
# class CodonBPETokenizer(Tokenizer):
class CodonBPETokenizer:
    """To be used at inference time for convenient DNA sequence encoding/decoding."""

    def __call__(self, text: Union[str, List[str]], **kwargs) -> BatchEncoding:
        """Convert the input DNA sequence (no spaces) to codons and then to bytes."""
        if isinstance(text, str):
            text = group_and_contextualize(text)
        else:
            text = [group_and_contextualize(t) for t in text]
        return super().__call__(text, **kwargs)

    def decode(*args, **kwargs) -> str:
        text = super().decode(*args, **kwargs)
        return "".join(CHAR_TO_CODON.get(c, "") for c in text)


def build_tokenizer(
    corpus_iterator,
    vocab_size=50_257,
    add_bos_eos: bool = True,
    max_length: int = 1024,
    save: bool = False,
    tokenzier_save_name: str = "cpe_tokenizer",
):
    special_tokens = {
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }

    bos_index = 5
    eos_index = 6

    # Define tokenizer
    tokenizer = Tokenizer(models.BPE())

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        max_length=max_length,
    )

    print("Training tokenizer")
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    # Add post-processor
    # trim_offsets=True will ignore spaces, false will leave them in
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    if add_bos_eos:
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", bos_index), ("[EOS]", eos_index)],
        )

    # Add a decoder
    tokenizer.decoder = decoders.ByteLevel()

    # save the tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, **special_tokens
    )
    if save:
        wrapped_tokenizer.save_pretrained(tokenzier_save_name)

    return wrapped_tokenizer


if __name__ == "__main__":
    sequence_file = Path()
    start = time.time()
    sequence_file = "/home/couchbucks/Downloads/all_fasta_files/training/GCA_000977415.2_Sc_YJM1385_v1_genomic_extracted_sequences.fasta"
    sequences = any_file_fasta_reader(sequence_file)
    sequences = [group_and_contextualize(seq.upper()) for seq in sequences]
    print(sequences[0])
    tokenizer = build_tokenizer(sequences, vocab_size=100)
    print("Tokenizer build time:", time.time() - start)
    print(tokenizer.vocab)
