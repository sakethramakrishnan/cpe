import glob
import queue
import re
import threading
import time
from pathlib import Path
from typing import List, Union

from tokenizers import Tokenizer, decoders, models, processors, trainers
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerFast

PathLike = Union[str, Path]


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


BASES = ["A", "T", "C", "G", "a", "t", "c", "g"]


def check_bases(seq):
    """Check that each of the letters in each sequence is of the set{'A', 'T', 'C', 'G'}"""
    return not any(x not in BASES for x in seq)


def replace_invalid_codons(codon_list):
    for idx, codon in enumerate(codon_list):
        if not check_bases(codon):
            codon_list[idx] = "XXX"
    return codon_list


def truncate_codon_sequence(sequence):
    """If the sequence is not evenly divisible by 3, then we take off %3 bases from the end"""
    remainder = len(sequence) % 3
    if remainder != 0:
        sequence = sequence[:-remainder]
    return sequence


def seq_to_codon_list(seq: str) -> List[str]:
    """split the sequence string into strings of len 3"""
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def group_and_contextualize(seq: str, k: int = 3):
    return "".join(CODON_TO_CHAR.get(seq[i : i + k], "") for i in range(0, len(seq), k))


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


def read_fasta_only_seq(fasta_file: PathLike) -> List[str]:
    """Reads fasta file sequences without description tag."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return lines[1::2]


def fasta_corpus_iterator(fasta_folder: Union[Path, List[Path]]):
    """Iterates over a set of fasta files one sequence at a time.

    Note: Does not skip any sequences, if a sequence length is not
    divisible by 3, it is truncated.
    """
    # fasta_files = []
    # fasta_folder = [fasta_folder] if isinstance(fasta_folder, Path) else fasta_folder
    # for p in Path(fasta_folder).glob("*.fasta"):
    #     fasta_files.extend(p)
    print("Reading sequences")
    for file in glob.iglob(f"{fasta_folder}/*"):
        for sequence in tqdm(SequenceReader(file)):
            return sequence


def build_tokenizer(
    corpus_iterator,
    vocab_size=50_257,
    add_bos_eos: bool = True,
    max_length: int = 1024,
    save: bool = False,
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
        wrapped_tokenizer.save_pretrained("test_tokenizer")

    return wrapped_tokenizer


if __name__ == "__main__":
    sequence_file = Path()
    start = time.time()
    sequences = SequenceReader(sequence_file)
    # sequences = fasta_corpus_iterator(sequence_file)
    tokenizer = build_tokenizer(sequences, vocab_size=40_000)
    print("Tokenizer build time:", time.time() - start)
