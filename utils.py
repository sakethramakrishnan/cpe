import glob
import queue
import threading
from operator import itemgetter
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]

import os
import random
import re
import time
from collections import Counter, defaultdict

import tqdm
from Bio.SeqUtils import gc_fraction
from Bio.Seq import translate
from pydantic import BaseModel


class Sequence(BaseModel):
    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""


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

BASES = ["A", "T", "C", "G", "a", "t", "c", "g"]


def group_and_contextualize(seq: str, num_char_per_token: int, convert_to_aa: bool, tokenizer_type: str) -> str:
    """ 
    Prepares a sequence to be tokenized by the given tokenizer
    Note: all tokenizers require spaces between each character
    
    ape, npe, protein_alphabet, and dna_wordlevel should be k = 1
    cpe and codon_wordlevel should be k = 3

    Args:
        seq (str): one sequence of DNA nucleotides or amino acids
        k (int): the 
        tokenizer_type (str): choices=['ape_tokenizer', 'npe_tokenizer', 'cpe_tokenizer', 'codon_wordlevel', 'dna_wordlevel', 'protein_alphabet_wordlevel']

    Returns:
        str: a string of the grouped, separated, and/or contextualized sequences
    """
    seq.replace(" ", "")
    if tokenizer_type == 'cpe_tokenizer':
        return " ".join(CODON_TO_CHAR.get(seq[i : i + num_char_per_token], "") for i in range(0, len(seq), num_char_per_token))
    
    if convert_to_aa:
        substrings = [translate(seq)[i:i + num_char_per_token] for i in range(0, len(seq), num_char_per_token)]
    else:
        substrings = [seq[i:i + num_char_per_token] for i in range(0, len(seq), num_char_per_token)]
    return ' '.join(substrings)


def read_fasta(fasta_file: PathLike) -> List[Sequence]:
    """Reads fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return [(seq, tag) for seq, tag in zip(lines[1::2], lines[::2])]


def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))


def filter_sequences_by_gc(dna_sequences: List[List[str]]) -> List[str]:
    """all sequences that have a GC content of 0% or 100%, we eliminate from the list of sequences"""
    refined_sequences = []
    for i, seq in enumerate(dna_sequences):
        gc_content = gc_fraction(seq)
        if (
            gc_content > 0.0
            and gc_content < 100.0
            and len(seq) >= 3
        ):
            refined_sequences.append(dna_sequences[i])

    return refined_sequences


def parse_sequence_labels(sequences: List[Sequence]) -> List[str]:
    pattern = r"gbkey=([^;]+)"
    matches = [re.search(pattern, seq[1]) for seq in sequences]
    labels = [match.group(1) if match else "" for match in matches]
    return labels


def find_invalid_seqs(dna_sequences: List[List[str]]) -> List[str]:
    """all sequences that have a GC content of 0% or 100%, we eliminate from the list of sequences"""
    bad_sequences = []
    bad_indices = []
    for i, seq in enumerate(dna_sequences):
        gc_content = gc_fraction(seq)
        if (
            gc_content == 0.0
            or gc_content == 100.0
            or not check_bases(seq)
            and len(seq) >= 3
        ):
            bad_sequences.append(dna_sequences[i])
            bad_indices.append(i)

    return bad_sequences, bad_indices


def preprocess_data(
    sequences: List[Sequence], labels: List[Sequence], per_of_each_class=1.0
):
    # Note: This function modifies sequences and labels
    # Filter out any outlier labels
    valid_labels = set(
        ["mRNA", "tRNA", "RNA", "exon", "misc_RNA", "rRNA", "CDS", "ncRNA"]
    )
    valid_inds_labels = [i for i, label in enumerate(labels) if label in valid_labels]
    valid_inds_sequences = filter_sequences_by_gc_and_bases(sequences)
    valid_inds = intersection(valid_inds_labels, valid_inds_sequences)
    sequences = [sequences[ind] for ind in valid_inds]
    labels = [labels[ind] for ind in valid_inds]

    label_group = defaultdict(list)
    for i, label in enumerate(labels):
        label_group[label].append(i)

    class_lens = dict(Counter(labels))
    for key in class_lens:
        class_lens[key] = round(class_lens[key] * per_of_each_class)
    print(class_lens)
    smallest_class, min_class_size = min(class_lens.items(), key=itemgetter(1))
    # min_class_size = class_lens[smallest_class]
    print(f"Smallest class: {smallest_class} with {min_class_size} examples")

    sampled_inds = []
    for label, inds in label_group.items():
        sampled_inds.extend(random.sample(inds, k=min_class_size))

    sequences = [sequences[ind] for ind in sampled_inds]
    labels = [labels[ind] for ind in sampled_inds]
    print(
        str(len(sequences))
        + ": number of total sequences; even split between CDS, ncRNA, tRNA, mRNA, rRNA"
    )

    return sequences, labels


def gc_content(seqs: List[str]) -> List[float]:
    """Given a list of DNA sequences, return each sequence's GC content.

    Parameters
    ----------
    seqs : List[str]
        A list of DNA sequences.

    Returns
    -------
    List
        GC content of each DNA sequence.
    """
    return [gc_fraction(seq) for seq in seqs]


def check_bases(seq):
    """Check that each of the letters in each sequence is of the set{'A', 'T', 'C', 'G'}"""
    return not any(x not in BASES for x in seq)


def replace_unk(codon_list):
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


def format_seq(seq: str) -> str:
    seq = truncate_codon_sequence(seq)
    seq = seq_to_codon_list(seq.upper())
    seq = replace_unk(seq)
    return seq


def get_label_dict(labels: List[str]):
    """
    label_dict: a dict where
    Key (str): Value(int) is each_category_of_label:a_corresponding_number_between_0_and_len(label_categories)
    len = len(label_categories)
    """
    all_possible_labels = set(labels)
    label_dict = {}
    for i, label in enumerate(all_possible_labels):
        label_dict[label] = i

    return label_dict


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