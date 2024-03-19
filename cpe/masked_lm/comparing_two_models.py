import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import wandb
import yaml
from transformers import (
    BertForMaskedLM,
    GPTNeoXForCausalLM,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from dataset import FastaDataset, GenSLMColatorForLanguageModeling
import evaluate
from utils import cpe_decode, get_aligned_seqs, get_aligned_seqs_biopython
import numpy as np

MODEL_DISPATCH = {
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "BertForMaskedLM": BertForMaskedLM,
    "neox": GPTNeoXForCausalLM,
    "bert": BertForMaskedLM,
    "GPT": GPTNeoXForCausalLM,
    "gpt": GPTNeoXForCausalLM
}

BPE_TOKENIZERS = ['ape_tokenizer', 'bpe_tokenizer', 'cpe_tokenizer', 'npe_tokenizer']


