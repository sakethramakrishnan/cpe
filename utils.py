from operator import itemgetter

from typing import Any, Dict, List, Optional, Set, Type, Union, Tuple


from pathlib import Path
PathLike = Union[str, Path]

import re
from pydantic import BaseModel
import random

from collections import Counter, defaultdict

from Bio.SeqUtils import GC

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
        (seq, tag) for seq, tag in zip(lines[1::2], lines[::2])
    ]

def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

def filter_sequences_by_gc(dna_sequences):
    ''' all sequences that have a GC content of 0% or 100%, we eliminate from the list of sequences '''
    valid_inds = []
    for i, sequence in enumerate(dna_sequences):
        gc_content = GC(sequence[0])
        if gc_content > 0. and  gc_content < 100. and len(sequence[0])>=3:
            valid_inds.append(i)
    return valid_inds

def parse_sequence_labels(sequences: List[Sequence]) -> List[str]:
    pattern = r'gbkey=([^;]+)'
    matches = [re.search(pattern, seq[1]) for seq in sequences]
    labels = [match.group(1) if match else "" for match in matches]
    return labels


def preprocess_data(sequences: List[Sequence], labels: List[Sequence], per_of_each_class=1.0):
    # Note: This function modifies sequences and labels
    # Filter out any outlier labels
    valid_labels = set(['mRNA', 'tRNA', 'RNA', 'exon', 'misc_RNA', 'rRNA', 'CDS', 'ncRNA'])
    valid_inds_labels = [i for i, label in enumerate(labels) if label in valid_labels]
    valid_inds_sequences = filter_sequences_by_gc(sequences)
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
    print(str(len(sequences)) + ": number of total sequences; even split between CDS, ncRNA, tRNA, mRNA, rRNA")

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

def get_label_dict(labels: List[str]):
    '''
    label_dict: a dictwhere
    Key (str): Value(int) is each_category_of_label:a_corresponding_number_between_0_and_len(label_categories)
    len = len(label_categories)
    '''
    all_possible_labels = set(labels)
    label_dict = {}
    for i, label in enumerate(all_possible_labels):
        label_dict[label] = i

    return label_dict
