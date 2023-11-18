import queue
import threading
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from genslm.utils import Sequence, read_fasta, read_fasta_only_seq
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# Assign a unique character to each codon so that we can use it as an
# input token to a BPE tokenizer. This implements a codon-pair encoding.
CODON_CHAR = {
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
    "XXX": "*"
}


def group_and_contextualize(seq: str, k: int = 3):
    return "".join(CODON_CHAR.get(seq[i : i + k], "") for i in range(0, len(seq), k))


class SequenceReader:
    def __init__(self, fasta_file: Path) -> None:
        self.finished = False
        self.queue = queue.Queue()
        self.thread = threading.Thread(
            target=self.read_fasta, args=(fasta_file,), daemon=True
        )
        self.thread.start()

    def read_fasta(self, fasta_file: Path):
        """Reads sequences one by one from a fasta file and yields the result"""
        current_sequence = []
        with open(fasta_file, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    # Edge case at the start of the file
                    if current_sequence:
                        sequence = group_and_contextualize(
                            "".join(current_sequence).upper()
                        )
                        self.queue.put(sequence)
                        current_sequence = []
                else:
                    current_sequence.append(line)

            # Edge case for the final sequence
            if current_sequence:
                sequence = group_and_contextualize("".join(current_sequence).upper())
                self.queue.put(sequence)

        self.finished = True

    def __iter__(self):
        i = 0
        while not (self.finished and self.queue.empty()):
            try:
                yield self.queue.get_nowait()
            except queue.Empty:
                time.sleep(1)  # Wait a second for the queue to fill
                continue

            i += 1
            if i % 50000 == 0:
                print(f"Qsize: {self.queue.qsize()}")


def fasta_corpus_iterator(fasta_file: Union[Path, List[Path]]):
    """Iterates over a set of fasta files one sequence at a time.

    Note: Does not skip any sequences, if a sequence length is not
    divisible by 3, it is truncated.
    """
    fasta_file = [fasta_file] if isinstance(fasta_file, Path) else fasta_file
    print("Reading sequences")
    for file in fasta_file:
        for sequence in tqdm(SequenceReader(file)):
            yield sequence


def build_tokenizer(corpus_iterator, vocab_size=50_257, add_bos_eos: bool = True):
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
        vocab_size=vocab_size, special_tokens=list(special_tokens.values())
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

    return wrapped_tokenizer


if __name__ == "__main__":
    sequence_file = Path(
        "/lambda_stor/homes/khippe/genslm_foundation/genome_data/curriculum_datasets/homology-90-50/codon_homology/pgfam_30k_all.ffn"
    )
    start = time.time()
    sequences = SequenceReader(sequence_file)
    # sequences = fasta_corpus_iterator(sequence_file)
    tokenizer = build_tokenizer(sequences)
    print("Tokenizer time:", time.time() - start)
    tokenizer.save_pretrained("pgfam_30k_all_vocab_50257")
