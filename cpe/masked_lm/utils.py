import glob
import queue
import threading
from operator import itemgetter
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]

import random
import re
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from Bio.Seq import translate
from Bio.SeqUtils import GC
from pydantic import BaseModel
#from sequence_alignment import GlobalAlignment, LocalAlignment
from Bio import pairwise2

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

CHAR_TO_CODON = {v: k for k, v in CODON_TO_CHAR.items()}


def group_and_contextualize_cpe_training(seq: str, k: int = 3):
    return "".join(CODON_TO_CHAR.get(seq[i : i + k]) for i in range(0, len(seq), k))


def group_and_contextualize(
    seq: str, num_char_per_token: int, convert_to_aa: bool, tokenizer_type: str
) -> str:
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
    if tokenizer_type in ['npe_tokenizer', 'ape_tokenizer', 'cpe_tokenizer']:
        if tokenizer_type == "cpe_tokenizer":
            try:
                return "".join(
                    CODON_TO_CHAR[seq[i : i + num_char_per_token]]
                    for i in range(0, len(seq), num_char_per_token)
                )
            except KeyError:
                raise ValueError(f"Invalid sequence during codon to char:\n{seq}")
        elif tokenizer_type == 'npe_tokenizer':
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        elif tokenizer_type == 'ape_tokenizer':
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        else:
            raise ValueError(f"Invalid tokenizer type: {tokenizer_type}. Must be one of: ['ape_tokenizer', 'npe_tokenizer', 'cpe_tokenizer', 'codon_wordlevel', 'dna_wordlevel', 'protein_alphabet_wordlevel']")
        return "".join(substrings)

    else:
        if convert_to_aa:
            # this assumes you have the sequences already translated in a .fasta file
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        else:  # Nucleotide case
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        return " ".join(substrings)


def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))


def filter_sequences_by_gc(dna_sequences: List[List[str]]) -> List[str]:
    """all sequences that have a GC content of 0% or 100%, we eliminate from the list of sequences"""
    refined_sequences = []
    for seq in dna_sequences:
        gc_content = GC(seq)
        if gc_content > 0.0 and gc_content < 100.0 and len(seq) >= 3:
            refined_sequences.append(seq)

    return refined_sequences


def parse_sequence_labels(sequences: List[Sequence]) -> List[str]:
    pattern = r"gbkey=([^;]+)"

    matches = [re.search(pattern, seq.tag) for seq in sequences]
    
    labels = [match.group(1) if match else "" for match in matches]
    return labels

def extract_gbkey(tag):
    match = re.search(r'gbkey=([^;]+)', tag)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"One of the sequences does not have a proper tag: {tag}")

def find_invalid_seqs(dna_sequences: List[List[str]]) -> List[str]:
    """all sequences that have a GC content of 0% or 100%, we eliminate from the list of sequences"""
    bad_sequences = []
    bad_indices = []
    for i, seq in enumerate(dna_sequences):
        gc_content = GC(seq)
        if (
            gc_content == 0.0
            or gc_content == 100.0
            or not check_bases(seq)
            and len(seq) >= 3
        ):
            bad_sequences.append(dna_sequences[i])
            bad_indices.append(i)

    return bad_sequences, bad_indices


def preprocess_data_dna(
    sequences: List[Sequence], labels: List[Sequence], per_of_each_class=1.0
):
    # Note: This function modifies sequences and labels
    # Filter out any outlier labels
    valid_labels = set(
        ["mRNA", "tRNA", "RNA", "exon", "misc_RNA", "rRNA", "CDS", "ncRNA"]
    )
    valid_inds_labels = [i for i, label in enumerate(labels) if label in valid_labels]

    sequences = [sequences[ind] for ind in valid_inds_labels]
    labels = [labels[ind] for ind in valid_inds_labels]

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
    return [GC(seq) for seq in seqs]


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
    if Path(fasta_file).is_file():
        fasta_files = [fasta_file]
    else:
        fasta_files = Path(fasta_file).glob("*.fasta")

    sequences = []
    for p in fasta_files:
        sequences.extend(read_fasta_only_seq(p))

    return sequences


class Sequence(BaseModel):
    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""


def read_fasta(fasta_file: PathLike) -> List[Sequence]:
    """Reads fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag) for seq, tag in zip(lines[1::2], lines[::2])
    ]


def any_file_fasta_reader_whole(fasta_file: PathLike) -> List[str]:
    if Path(fasta_file).is_file():
        fasta_files = [fasta_file]
    else:
        fasta_files = Path(fasta_file).glob("*.fasta")

    sequences = []
    for p in fasta_files:
        sequences.extend(read_fasta(p))

    return sequences


def write_fasta(
    sequences: Union[Sequence, List[Sequence]], fasta_file: PathLike, mode: str = "w"
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f">{seq.tag}\n{seq.sequence}\n")


def make_str_div(seq, n):
    remainder = len(seq) % n
    # print(remainder)
    if remainder != 0:
        seq = seq[:-remainder]

    return seq


def make_perfect_fasta(
    current_fasta_files: PathLike, write_fasta_file: str, mode: str = "w"
):
    """Create a new fasta file with the 'proper' sequences"""

    seqs_and_tags = any_file_fasta_reader_whole(current_fasta_files)
    sequences = [make_str_div(seq.sequence.upper(), 3) for seq in seqs_and_tags]
    tags = [seq.tag for seq in seqs_and_tags]

    valid_inds = []

    refined_seqs = []

    for i, sequence in enumerate(sequences):
        if (
            GC(sequence) > 0.0
            and GC(sequence) < 100.0
            and len(sequence) >= 3
        ):
            codon_list = seq_to_codon_list(sequence)
            replaced_codon_list = replace_invalid_codons(codon_list)
            refined_seqs.append("".join(replaced_codon_list))
            valid_inds.append(i)

    refined_tags = [tags[i] for i in valid_inds]

    with open(write_fasta_file, mode) as f:
        for seq, tag in zip(refined_seqs, refined_tags):
            f.write(f">{tag}\n{seq}\n")

def convert_dna_to_aa(
    current_fasta_files: PathLike, write_fasta_file: str, mode: str = "w"
):
    seqs_and_tags = any_file_fasta_reader_whole(current_fasta_files)
    sequences = [make_str_div(seq.sequence.upper(), 3) for seq in seqs_and_tags]
    tags = [seq.tag for seq in seqs_and_tags]
    
    aa_seq = [translate(seq) for seq in sequences]
    
    with open(write_fasta_file, mode) as f:
        for seq, tag in zip(aa_seq, tags):
            f.write(f">{tag}\n{seq}\n")

def plot_tokenized_len_hist(tokenizer, sequences: list):
    tokenized_seqs = tokenizer(
        sequences,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    data = {
        "input_ids": tokenized_seqs.input_ids.tolist(),
        "attention_mask": tokenized_seqs.attention_mask.tolist(),
    }

    dataset = Dataset.from_dict(data)
    print(dataset)

    tokenized_lens = [sum(elem["attention_mask"]) for elem in dataset]
    pd.DataFrame(tokenized_lens).describe()

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Plotting the histogram
    plt.hist(tokenized_lens, bins=50, color="blue", alpha=0.7)

    # Setting title and labels
    plt.title(current_tokenizer_name)
    plt.xlabel("Tokenized Length")
    plt.ylabel("Frequency")

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_vocab_len_hist(tokenizer):
    vocab_lens = [len(elem) for elem in tokenizer.get_vocab().keys()]

    df = pd.DataFrame(vocab_lens)
    print(df.describe())

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Plotting the histogram
    plt.hist(vocab_lens, bins=50, color="blue", alpha=0.7)
    plt.yscale("log")

    # Setting title and labels
    plt.title("Vocab lengths")
    plt.xlabel("Sequence Motif Length")
    plt.ylabel("Frequency")

    # Display the plot
    plt.tight_layout()
    plt.show()

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in codon_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def plot_vocab_seq_lens(tokenizer, sequences):
    # plot of vocab sequence lengths
    
    tokenized_seqs = tokenizer(
        sequences,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    counts = defaultdict(int)
    for elem in tqdm(tokenized_seqs.input_ids):
        for id in elem[1:-1]:
            id_str = tokenizer.decode([id])
            counts[id_str] += 1
    # remove special tokens from being counted
    for val in special_tokens.values():
        if val in counts:
            del counts[val]

    vocab_occs = list(counts.values())
    vocab_occs_filtered = [v for v in vocab_occs if v < 10000]

    import pandas as pd

    df = pd.DataFrame(vocab_occs_filtered)
    print(df.describe())

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Plotting the histogram
    plt.hist(vocab_occs_filtered, bins=100, color="blue", alpha=0.7)
    plt.yscale("log")

    # Setting title and labels
    plt.title("Vocab token occurences")
    plt.xlabel("??")
    plt.ylabel("Frequency")

    # Display the plot
    plt.tight_layout()
    plt.show()
    

def get_aligned_seqs(seq1, seq2):
    # Save results to database and print them

    ga_align1, ga_align2, ga_alignment_score, la_align1, la_align2, la_alignment_score = get_results(seq1, seq2)

    return ga_align1, ga_align2, ga_alignment_score, la_align1, la_align2, la_alignment_score


# Save alignment results from all combinations of sequences to database
# (And print them if program is being run as CLI tool)
def get_results(seq1, seq2):

    #for seq1, seq2 in combinations(seqs, 2):

        ga = GlobalAlignment((seq1, seq2))
        la = LocalAlignment((seq1, seq2))

        ga_align1, ga_align2 = ga.align
        la_align1, la_align2 = la.align

        return ga_align1, ga_align2, int(ga.alignment_score), la_align1, la_align2, int(la.alignment_score)




def get_aligned_seqs_biopython(seq1, seq2):
    alignments_global = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)

    seq1_global = alignments_global[0].seqA # the alignments are in a 1 length list
    seq2_global = alignments_global[0].seqB
    global_score = alignments_global[0].score

    alignments_local = pairwise2.align.localxx(seq1, seq2, one_alignment_only=True)

    seq1_local = alignments_local[0].seqA # the alignments are in a 1 length list
    seq2_local = alignments_local[0].seqB
    local_score = alignments_local[0].score

    return seq1_global, seq2_global, global_score, seq1_local, seq2_local, local_score


def cpe_decode(seq, *args, **kwargs) -> str:
        return "".join(CHAR_TO_CODON.get(c, "") for c in seq)
    
def most_common_token(tokenizer, sequences):
    tokenized_seqs = tokenizer(
        sequences,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    counts = defaultdict(int)
    for elem in tqdm(tokenized_seqs.input_ids):
        for id in elem[1:-1]:
            id_str = tokenizer.decode([id])
            counts[id_str] += 1
    # remove special tokens from being counted
    for val in special_tokens.values():
        if val in counts:
            del counts[val]

    vocab_occs = list(counts.values())
    common_token_idx = np.argmax(vocab_occs)
    common_token = list(counts.keys())[common_token_idx]
    print(f"{vocab_occs[common_token_idx]} occurences", common_token, decode_grouped_context(
        common_token
    ))
    
    
#make_perfect_fasta("/home/couchbucks/Documents/saketh/cpe/data/datasets/all_fungi_raw.fasta", "fungi_refined.fasta")


def build_bpe_tokenizer(
    corpus_iterator: list[str], # consists of a list of genetic sequence to train the bpe tokenizer on
    vocab_size: int, # number of tokens in the vocabulary
    tokenizer_type: str, # choices: ['ape_tokenizer', 'npe_tokenizer', 'cpe_tokenizer']
    add_bos_eos: bool = True,
    save: bool = False,
    initial_alphabet: list[str] = None, # tokenizer must have these chars: ex. ['A', 'T', 'C', 'G'] for npe_tokenizer; uncomment the line with initial_alphabet
    save_name: str = '',
    cpe_translated: bool = False # whether or not the input corpus is already translated into CPE language
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
    tokenizer = Tokenizer(BPE(unk_token=special_tokens["unk_token"]))

    if tokenizer_type == "cpe_tokenizer":

        sequences = []
        for x in corpus_iterator:
            sequences.extend(x)
        try:
            corpus_iterator = [
                group_and_contextualize(
                    seq, num_char_per_token=3, convert_to_aa=False, tokenizer_type='cpe_tokenizer'
                )
                for seq in sequences
            ]

        except: # for when the sequences are already translated into CPE language
            pass

        initial_alphabet = list(CODON_TO_CHAR.values())

        tokenizer.pre_tokenizer = Digits(individual_digits=False)
    else:
        if tokenizer_type == 'ape_tokenizer':
            initial_alphabet = ['L','K', 'P','I','E','V','G','N','D','S','W','M','Y','R','C','A','T','F','_','Q','H','X']
        elif tokenizer_type == 'npe_tokenizer':
            initial_alphabet = ['A', 'T', 'C', 'G']


        tokenizer.pre_tokenizer = Whitespace()

    print(vocab_size)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        initial_alphabet=initial_alphabet # have an initial alphabet just in case of sequences without a particular base or amino acid :)
    )

    print("Training tokenizer")

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    # Add post-processor
    # trim_offsets=True will ignore spaces, false will leave them in
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
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

        wrapped_tokenizer.save_pretrained(save_name)

    print(f"Returning tokenizer with vocab_size = {tokenizer.get_vocab_size()}")

    return wrapped_tokenizer


def pad_nucleotides(seq1, seq2, pad_char="p"):
    # Find the maximum length between the sequences
    max_length = max(len(seq1), len(seq2))
    # Pad the sequences with "x" to make them the same length
    seq1 = seq1.ljust(max_length, pad_char)
    seq2 = seq2.ljust(max_length, pad_char)
    
    return seq1, seq2
    
    
def get_num_correct_punish(prediction, ground_truth):
    # Zip the sequences, including positions where either has "-"
    # Calculate accuracy
    correct_count = 0
    total_count = 0
    for p, g in zip(prediction, ground_truth):
        total_count += 1
        if p == g:
            correct_count += 1
    return correct_count, total_count



def get_num_correct(prediction, ground_truth):
    # Zip the sequences, excluding positions where either has "-"
    zipped_sequences = zip(prediction, ground_truth)
    valid_pairs = [(p, g) for p, g in zipped_sequences if p != '-' and g != '-']
    # Calculate accuracy
    correct_count = sum(p == g for p, g in valid_pairs)
    total_count = len(valid_pairs)
    return correct_count, total_count