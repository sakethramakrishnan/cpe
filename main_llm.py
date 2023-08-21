import argparse
import time
import queue
import pandas as pd
import threading
from typing import Union, List, Generator
from tqdm import tqdm
from pathlib import Path
import sys
import bpe_tokenizer
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from genslm.utils import read_fasta_only_seq, read_fasta, Sequence
from transformers import AutoTokenizer
from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, PretrainedConfig
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

CODON_CHAR = {
    'TCG': "A", 'GCA': "B", 'CTT': "C", 'ATT': "D", 'TTA': "E", 'GGG': "F", 'CGT': "G",
    'TAA': "H", 'AAA': "I", 'CTC': "J", 'AGT': "K", 'CCA': "L", 'TGT': "M", 'GCC': "N",
    'GTT': "O", 'ATA': "P", 'TAC': "Q", 'TTT': "R", 'TGC': "S", 'CAC': "T", 'ACG': "U",
    'CCC': "V", 'ATC': "W", 'CAT': "X", 'AGA': "Y", 'GAG': "Z", 'GTG': "a", 'GGT': "b",
    'GCT': "c", 'TTC': "d", 'AAC': "e", 'TAT': "f", 'GTA': "g", 'CCG': "h", 'ACA': "i",
    'CGA': "j", 'TAG': "k", 'CTG': "l", 'GGA': "m", 'ATG': "n", 'TCT': "o", 'CGG': "p",
    'GAT': "q", 'ACC': "r", 'GAC': "s", 'GTC': "t", 'TGG': "u", 'CCT': "v", 'GAA': "w",
    'TCA': "x", 'CAA': "y", 'AAT': "z", 'ACT': "0", 'GCG': "1", 'GGC': "2", 'CTA': "3",
    'AAG': "4", 'AGG': "5", 'CAG': "6", 'AGC': "7", 'CGC': "8", 'TTG': "9", 'TCC': "!",
    'TGA': "@"
}



def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for BERT Model")

    # Model hyperparameters
    parser.add_argument('--model_name', type=str, default='BertForMaskedLM', choices=['bert_3m', 'bert_33m', 'bert_330m'], help='Name of the Hugging Face BERT model')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Path to a pre-trained BERT model checkpoint')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum input sequence length')
    parser.add_argument('--truncation', type=int, default=True, help='Truncating sequences to fit within a specified maximum length')

    # filepath to fasta files:
    parser.add_argument('--fasta_filepath', type=str, default=None, help='The filepath to the .fasta file with sequences')
    parser.add_argument('--fasta_folder_filepath', type=str, default=None, help='The filepath to the FOLDER with many .fasta files')

    # Tokenizer hyperparameters
    parser.add_argument('--tokenizer_checkpoint', type=str, default=None, help='Path to a pre-trained tokenizer checkpoint')
    parser.add_argument('--padding', type=str, default='max_length', choices=['longest', 'max_length', 'do_not_pad'], help='Whether to pad the inputs')
    parser.add_argument('--vocab_size', type=int, default=50_257, help='The number of elements in the vocabulary of the tokenizer')

    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--per_device_train_batch_size', default=64, type=int, help='Per-GPU training batch-size : number of distinct sequences loaded on one GPU during training.')
    parser.add_argument('--per_device_eval_batch_size', default=64, type=int,
                        help='Per-GPU evaluation batch-size : number of distinct sequences loaded on one GPU during testing.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Percentage of dataset reserved for testing, expressed as a decimal.')


    # Learning hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon parameter for Adam optimizer')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw""")

    parser.add_argument('--eval_metric', type=str, default='accuracy', choices=['accuracy', 'bertscore', 'bleu', 'bleurt',
                                                                                'cer', 'comet', 'coval', 'cuad',
                                                                                'f1', 'gleu', 'glue', 'indic_glue',
                                                                                'matthews_correlation', 'meteor',
                                                                                'pearsonr', 'precision', 'recall', 'rouge',
                                                                                'sacrebleu', 'sari', 'seqeval', 'spearmanr',
                                                                                'squad', 'squad_v2', 'super_glue', 'wer',
                                                                                'wiki_split', 'xnli'], help='The type of '
                                                                                                            'evaluation metric '
                                                                                                            'provided by HuggingFace')
    parser.add_argument('--output_dir', type=str, default='bpe_llm_out', help='Path where to save visualizations.')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='which learning rate scheduler to use')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', choices=['no', 'epoch', 'steps'], help='Whether to evaluate the model after epochs, or steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='If evaluation_strategy=steps, after x steps in training, evaluate the model')
    parser.add_argument('--logging_strategy', type=str, default='steps', choices=['no', 'epoch', 'steps'],
                        help='Log after each epoch, steps, or not log at all')
    parser.add_argument('--logging_steps', type=int, default=100, help='If logging_strategy=steps, after x steps in training, log the model')
    parser.add_argument('--fp16', type=bool_flag, default=True, help='Whether or not to use half precision for training. '
                                                                'Improves training time and memory requirements, '
                                                                'but can provoke instability and slight decay of performance. '
                                                                'NOTE: CAN ONLY BE USED ON CUDA DEVICES')
    parser.add_argument('--mlm', type=bool_flag, default=True, help='Whether or not the data_collator can use Masked language Modeling')
    parser.add_argument('--save_steps', type=int, default=500, help='Number of updates steps before two checkpoint saves if save_strategy="steps"')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of steps used for a linear warmup from 0 to learning_rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass')


    args = parser.parse_args()
    return args

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_sequences():
    if args.fasta_filepath != None and args.fasta_folder_filepath != None:
        sys.exit("Kindly use either a single .fasta file or a folder with many .fasta files in it; do not input both as arguments")
    elif args.fasta_filepath != None:
        sequences = bpe_tokenizer.read_fasta_only_seq(args.fasta_filepath)
        sequences = [bpe_tokenizer.group_and_contextualize(seq) for seq in sequences]
    elif args.fasta_folder_filepath != None:
        sequences = bpe_tokenizer.fasta_corpus_iterator(args.fasta_folder_filepath)
        sequences = [bpe_tokenizer.group_and_contextualize(seq) for seq in sequences]

    return sequences

def get_tokenizer(sequence: Generator):

    special_tokens = {
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]"
    }
    bos_index = 5
    eos_index = 6

    if args.tokenizer_checkpoint:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_checkpoint)
        tokenizer.add_special_tokens(special_tokens)
    else:
        tokenizer = bpe_tokenizer.build_tokenizer(sequences, args.vocab_size)

    return tokenizer



# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

# Rest of your code...
from datasets import load_metric


def compute_metrics(p):
    metric = load_metric(args.eval_metric)
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids
    )



def get_dataset(sequences, tokenizer):
    tokenized_seqs = tokenizer(sequences, max_length=args.max_length, padding=args.padding, truncation=args.truncation,
                               return_tensors="pt")
    data = {
        "input_ids": tokenized_seqs.input_ids.tolist(),
        "attention_mask": tokenized_seqs.attention_mask.tolist()
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=args.test_size)

    return dataset
def get_model():

    if args.model_name == 'bert_3m':
        arch_path = Path('/cpe/architectures/bert/bert_3m.json')
        config = PretrainedConfig.from_json_file(arch_path)
        model = BertForMaskedLM(config)

    elif args.model_name == 'bert_33m':
        arch_path = Path('/cpe/architectures/bert/bert_33m.json')
        config = PretrainedConfig.from_json_file(arch_path)
        model = BertForMaskedLM(config)

    elif args.model_name == 'bert_330m':
        arch_path = Path('/cpe/architectures/bert/bert_330m.json')
        config = PretrainedConfig.from_json_file(arch_path)
        model = BertForMaskedLM(config)

    else:
        sys.exit('Please provide a valid model architecture in the "model_name" argument')


    return model

def get_optimizer():
    if args.optimizer == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            ep=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )

    else:
        sys.exit('Your optimizer is not one that our pipeline supports')

    return optimizer

def get_lr_scheduler(optimizer):
    if args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    else:
        sys.exit('Your learning rate scheduler is not one that our pipeline supports')
    return scheduler

def get_training_args():
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        fp16=args.fp16,
        push_to_hub=False,
    )

    return training_args

def train_model(model, train_args: TrainingArguments, tokenizer, dataset, data_collator):
    # Build model

    # Build trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    # train

    return trainer

if __name__ == "__main__":

    os.environ["WANDB_DISABLED"] = "true"
    args = get_args()
    sequences = get_sequences()
    tokenizer = get_tokenizer(sequences)

    from transformers import AutoTokenizer, AutoModel

    model = get_model()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=args.mlm)
    dataset = get_dataset(sequences, tokenizer)
    optimizer = get_optimizer()
    scheduler = get_lr_scheduler(optimizer)
    training_args = get_training_args()
    trainer = train_model(model, training_args, tokenizer, dataset, data_collator)

    trainer.train()


