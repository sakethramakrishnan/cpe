import time
from pathlib import Path
from typing import List, Union

from tokenizers import Tokenizer, decoders, models, processors, trainers
from tokenizers.processors import TemplateProcessing
# TODO: How to import any_file_fasta_reader from utils
# from cpe.utils import any_file_fasta_reader
from transformers import BatchEncoding, PreTrainedTokenizerFast
PathLike = Union[str, Path]
import os
import re

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
        raise ValueError("Kindly enter a filepath to a directory containing many .fasta files "
                         "or a filepath to a single .fasta file")

    return sequences_raw

def build_tokenizer(
    corpus_iterator,
    vocab_size=50_257,
    add_bos_eos: bool = True,
    max_length: int = 1024,
    save: bool = False,
    tokenzier_save_name: str = 'npe_tokenizer'
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
    sequence_file_training = '/home/couchbucks/Downloads/all_fasta_files/training'
    sequence_file_testing = '/home/couchbucks/Downloads/all_fasta_files/testing'
    sequences_training = any_file_fasta_reader(sequence_file)
    sequences_testing = any_file_fasta_reader(sequence_file_testing)
    sequences = sequences_testing + sequences_testing
    sequences = [seq for seq in sequences if len(seq)>=0]
    tokenizer = build_tokenizer(sequences, vocab_size=50_257, save=True)
    print("Tokenizer build time:", time.time() - start)