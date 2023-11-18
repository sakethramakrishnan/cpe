import functools
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from all_cluster_visualization import PlotClustersData
from Bio import SeqIO  # type: ignore[import]
from main_llm import get_model, get_sequences, get_tokenizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerFast

from utils import (
    gc_content,
    get_label_dict,
    parse_sequence_labels,
    preprocess_data,
    read_fasta,
)

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
}


class SequenceDataset(Dataset):  # type: ignore[type-arg]
    """Dataset initialized from a list of sequence strings."""

    def __init__(
        self,
        sequences: List[str],
        seq_length: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
        verbose: bool = True,
    ):
        self.batch_encodings = self.tokenize_sequences(
            sequences, tokenizer, seq_length, kmer_size, verbose
        )

    @staticmethod
    def tokenize_sequences(
        sequences: List[str],
        tokenizer: PreTrainedTokenizerFast,
        seq_length: int,
        kmer_size: int = 3,
        verbose: bool = True,
    ) -> List[BatchEncoding]:
        tokenizer_fn = functools.partial(
            tokenizer,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        batch_encodings = [
            tokenizer_fn(SequenceDataset.group_and_contextualize(seq, kmer_size))
            for seq in tqdm(sequences, desc="Tokenizing...", disable=not verbose)
        ]
        return batch_encodings

    @staticmethod
    def group_and_contextualize(seq: str, k: int = 3):
        return "".join(
            CODON_TO_CHAR.get(seq[i : i + k], "") for i in range(0, len(seq), k)
        )

    def __len__(self) -> int:
        return len(self.batch_encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch_encoding = self.batch_encodings[idx]
        # Squeeze so that batched tensors end up with (batch_size, seq_length)
        # instead of (batch_size, 1, seq_length)
        sample = {
            "input_ids": batch_encoding["input_ids"].squeeze(),
            "attention_mask": batch_encoding["attention_mask"],
        }
        return sample


# enter the fasta filepath to a fasta path:
fasta_path = ""
seqs_raw = read_fasta(fasta_path)

labels = parse_sequence_labels(seqs_raw)
sequences, labels = preprocess_data(seqs_raw, labels)
label_dict = get_label_dict(labels)

label_categories = set(labels)

# enter the checkpoint to the tokenizer:
tokenizer_checkpoint = ""

tokenizer = get_tokenizer(
    sequences, tokenizer_checkpoint=tokenizer_checkpoint, vocab_size=50_257
)

# TODO: see how to get the seq_seq_lengthlength from the args in main_llm:
seq_length = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(
    tokenizer=tokenizer, model_architecture="bert_3m", model_checkpoint=None
)
model.eval()

model.to(device)

dataset = SequenceDataset(sequences, seq_length, tokenizer)
dataloader = DataLoader(dataset, batch_size=64)

embeddings = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        outputs = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            output_hidden_states=True,
        )
        # outputs.hidden_states shape: (layers, batch_size, sequence_length, hidden_size)
        # Use the embeddings of the last layer
        emb = outputs.hidden_states[-1].detach().cpu().numpy()
        # Compute average over sequence length
        emb = np.mean(emb, axis=1)
        embeddings.append(emb)

# Concatenate embeddings into an array of shape (num_sequences, hidden_size)
embeddings = np.concatenate(embeddings)

# embeddings should be of size (N, hidden_size)
print(embeddings.shape, ": shape of embeddings")

tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)
label_array = np.array([label_dict[x] for x in labels])
gc_content_of_seqs = np.array(gc_content(sequences))

embedding_visualization = PlotClustersData(
    tsne_embeddings, label_array, gc_content_of_seqs, label_dict
)

# plotting all clusters colored with label

(
    plot_df_separate,
    hue_separate,
    plt_title,
) = embedding_visualization.separate_clusters_labels()
embedding_visualization.plot_clusters(plot_df_separate, hue_separate, plt_title)
plt.show()

# plotting all points colored with gc content
for x in range(len(label_categories)):
    (
        plot_df_separate_gc_coding,
        hue,
        plt_title,
    ) = embedding_visualization.separate_clusters_gc_content(label_mask=x)
    embedding_visualization.plot_clusters(plot_df_separate_gc_coding, hue, plt_title)
    plt.show()


# plotting all clusters colored with gc content

(
    plot_df_separate,
    hue_separate,
    plt_title,
) = embedding_visualization.plot_both_clusters_gc_content()
embedding_visualization.plot_clusters(plot_df_separate, hue_separate, plt_title)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, stratify=labels, random_state=1
)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
print(f"Model train accuracy: {clf.score(X_train, y_train)}")
print(f"Model test accuracy: {clf.score(X_test, y_test)}")
