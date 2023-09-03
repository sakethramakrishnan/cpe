import argparse

from typing import Optional, List, Dict, Any

from pathlib import Path
import sys
import bpe_tokenizer
import evaluate

import os
import numpy as np

import torch
import transformers
import datasets

from transformers import (
    BertForMaskedLM,
    AdamW,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    GPTNeoXForCausalLM,
    BatchEncoding
)

from datasets import Dataset
from torch.utils.data import random_split
from torch.utils.data import Dataset as TypeDataset
import re

def get_args():
    parser = argparse.ArgumentParser(description="Parameters for our model")

    # Model hyperparameters
    parser.add_argument('--model_architecture', type=str, default='bert_3m', help='Path of the Hugging Face model architecture JSON file')

    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to a pre-trained BERT model checkpoint')

    # filepath to fasta files:
    parser.add_argument('--fasta_path', type=Path, default=None, help='The filepath or folderpath to the .fasta file(s) with sequences')


    # Tokenizer hyperparameters
    parser.add_argument('--tokenizer_checkpoint', type=str, default=None,
                        help='Path to a pre-trained tokenizer checkpoint')
    parser.add_argument('--padding', type=str, default='max_length', choices=['longest', 'max_length', 'do_not_pad'],
                        help='Whether to pad the inputs')

    parser.add_argument('--vocab_size', type=int, default=50_257,
                         help='The number of elements in the vocabulary of the tokenizer')

    parser.add_argument('--max_length', type=int, default=1024, help='Maximum input sequence length')
    parser.add_argument('--truncation', type=int, default=True,
                        help='Truncating sequences to fit within a specified maximum length')

    # Learning hyperparameters
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon parameter for Adam optimizer')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw""")

    parser.add_argument('--eval_metric', type=str, default='accuracy',
                        choices=['accuracy', 'bertscore', 'bleu', 'bleurt',
                                 'cer', 'comet', 'coval', 'cuad',
                                 'f1', 'gleu', 'glue', 'indic_glue',
                                 'matthews_correlation', 'meteor',
                                 'pearsonr', 'precision', 'recall', 'rouge',
                                 'sacrebleu', 'sari', 'seqeval', 'spearmanr',
                                 'squad', 'squad_v2', 'super_glue', 'wer',
                                 'wiki_split', 'xnli'], help='The type of '
                                                             'evaluation metric '
                                                             'provided by HuggingFace')

    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='which learning rate scheduler to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of dataset reserved for testing, expressed as a decimal.')
    parser.add_argument('--mlm', type=bool_flag, default=True,
                        help='Whether or not the data_collator can use Masked language Modeling')

    # Arguments used for training:
    parser.add_argument('--output_dir', type=str, default='bpe_llm_out', help='Path where to save visualizations.')
    parser.add_argument('--per_device_train_batch_size', default=64, type=int,
                        help='Per-GPU training batch-size : number of distinct sequences loaded on one GPU during training.')
    parser.add_argument('--per_device_eval_batch_size', default=64, type=int,
                        help='Per-GPU evaluation batch-size : number of distinct sequences loaded on one GPU during testing.')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', choices=['no', 'epoch', 'steps'],
                        help='Whether to evaluate the model after epochs, or steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='If evaluation_strategy=steps, after x steps in training, evaluate the model')
    parser.add_argument('--logging_strategy', type=str, default='steps', choices=['no', 'epoch', 'steps'],
                        help='Log after each epoch, steps, or not log at all')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='If logging_strategy=steps, after x steps in training, log the model')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass')
    parser.add_argument('--num_train_epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of steps used for a linear warmup from 0 to learning_rate')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='The initial learning rate for AdamW optimizer')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Number of updates steps before two checkpoint saves if save_strategy="steps"')
    parser.add_argument('--fp16', type=bool_flag, default=True,
                        help='Whether or not to use half precision for training. '
                             'Improves training time and memory requirements, '
                             'but can provoke instability and slight decay of performance. '
                             'NOTE: CAN ONLY BE USED ON CUDA DEVICES')
    parser.add_argument('--push_to_hub', type=bool_flag, default=False,
                        help='Whether or not to push the model to the HuggingFace hub')

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

def get_sequences(fasta_path: str):
    fasta_path = Path(fasta_path)
    if fasta_path.is_file():
        sequences = bpe_tokenizer.read_fasta_only_seq(fasta_path)
        print(len(sequences))
    else:
        sequences = bpe_tokenizer.fasta_corpus_iterator(fasta_path)

    sequences = [bpe_tokenizer.group_and_contextualize(seq) for seq in sequences]
    return sequences


def get_tokenizer(sequences: Optional = None, tokenizer_checkpoint: Optional = None, vocab_size: Optional = None):
    if tokenizer_checkpoint:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_checkpoint)
    else:
        tokenizer = bpe_tokenizer.build_tokenizer(sequences, vocab_size)

    return tokenizer



def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.reshape(predictions.size)
    labels = labels.reshape(predictions.size)
    return metric.compute(predictions=predictions, references=labels)


def get_dataset(sequences: List[str], tokenizer: List[transformers.tokenization_utils_fast.PreTrainedTokenizerFast], max_length: int, padding: str, truncation: bool, test_size: float):
    tokenized_seqs = tokenizer(
        sequences,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt"
    )


    data = {
        "input_ids": tokenized_seqs.input_ids.tolist(),
        "attention_mask": tokenized_seqs.attention_mask.tolist()
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=test_size)

    return dataset

def get_model(tokenizer, model_architecture: Optional = None, model_checkpoint: Optional = None):
    if model_checkpoint:
        model = BertForMaskedLM.from_pretrained(Path(model_checkpoint))

    elif model_architecture:

        if model_architecture == 'bert_3m':
            arch_path = Path('architectures/bert/bert_3m.json')
            config = PretrainedConfig.from_json_file(arch_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = BertForMaskedLM(config)

        elif model_architecture == 'bert_33m':
            arch_path = Path('architectures/bert/bert_33m.json')
            config = PretrainedConfig.from_json_file(arch_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = BertForMaskedLM(config)

        elif model_architecture == 'bert_330m':
            arch_path = Path('architectures/bert/bert_330m.json')
            config = PretrainedConfig.from_json_file(arch_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = BertForMaskedLM(config)

        elif model_architecture == 'neox_3m':
            arch_path = Path('architectures/neox/neox_3m.json')
            config = PretrainedConfig.from_json_file(arch_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = GPTNeoXForCausalLM(config)

        elif model_architecture == 'neox_33m':
            arch_path = Path('architectures/neox/neox_33m.json')
            config = PretrainedConfig.from_json_file(arch_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = BertForMaskedLM(config)

        elif model_architecture == 'neox_330m':
            arch_path = Path('architectures/neox/neox_330m.json')
            config = PretrainedConfig.from_json_file(arch_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = BertForMaskedLM(config)

    else:
        sys.exit('Please provide a valid model architecture in the "model_architecture" argument')

    return model

def get_checkpoint_model(model_path: str):
    model = BertForMaskedLM.from_pretrained(Path(model_path))

    return model
def get_optimizer(optimizer: str, learning_rate: float, weight_decay: float):
    if optimizer == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            # ep=adam_epsilon,
            weight_decay=weight_decay,
        )
    else:
        sys.exit('Your optimizer is not one that our pipeline supports')

    return optimizer


def get_lr_scheduler(lr_scheduler: str, optimizer, num_train_epochs: int):
    if lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_epochs, eta_min=0)
    else:
        sys.exit('Your learning rate scheduler is not one that our pipeline supports')
    return scheduler


def get_training_args(
        output_dir: str,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        evaluation_strategy: str,
        eval_steps: int,
        logging_strategy: str,
        logging_steps: int,
        gradient_accumulation_steps: int,
        num_train_epochs: int,
        weight_decay: float,
        warmup_steps: int,
        learning_rate: float,
        save_steps: int,
        fp16: bool,
        push_to_hub: bool
                      ):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        save_steps=save_steps,
        fp16=fp16,
        push_to_hub=push_to_hub,
    )

    return training_args


def train_model(
        model,
        train_args: TrainingArguments,
        tokenizer: List[transformers.tokenization_utils_fast.PreTrainedTokenizerFast],
        #dataset: List[datasets.dataset_dict.DatasetDict],
        train_dataset,
        test_dataset,
        data_collator,
        device
):
    # Build model

    # Build trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,#.to(device),
        eval_dataset=test_dataset,#.to(device),
        compute_metrics=compute_metrics
    )

    # train

    # TODO: Save a checkpoint at the very end

    return trainer
def group_codons(seq: str) -> str:
    return " ".join(seq[i : i + 3] for i in range(0, len(seq), 3)).upper()

def read_fasta_only_seq(fasta_file: str) -> List[str]:
    """Reads fasta file sequences without description tag."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "")
        for seq in non_parsed_seqs
        for line in seq.split("\n", 1)
    ]
    return lines[1::2]


class FastaDataset(TypeDataset):
    def __init__(self, file_path: str) -> None:
        # Read the fasta file
        dna_sequenes = read_fasta_only_seq(file_path)
        # Preprocess the sequences into codons
        # TODO: We could also use an <unk> token (this would be better)
        self.sequences = [
            group_codons(seq) for seq in dna_sequenes if len(seq) % 3 == 0
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # Get the idx'th codon sequence
        # return {"codon": self.sequences[idx]}
        return self.sequences[idx]


class GenSLMCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Augment the underlying DataCollatorForLanguageModeling to handle
    multiple batch encoding inputs."""

    def __init__(self, train_mode: bool = False, **kwargs) -> None:
        self.train_mode = train_mode
        super().__init__(**kwargs)

    def tokenize(self, sequences: List[str]) -> BatchEncoding:
        return self.tokenizer(
            sequences,
            max_length=1024,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_special_tokens_mask=self.train_mode and self.mlm,
        )

    def torch_call(self, examples: List[Dict[str, str]]) -> Dict[str, Any]:
        # First, tokenize the batch
        #print(examples)
        batch = self.tokenize([e for e in examples])

        # We only need to mask tokens if we are training
        if not self.train_mode:
            return batch

        if self.mlm:
            # If special token mask has been preprocessed, pop it from the dict.
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"],
                special_tokens_mask=batch.pop("special_tokens_mask", None),
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    args = get_args()
    sequences = get_sequences(args.fasta_path)
    tokenizer = get_tokenizer(sequences, args.tokenizer_checkpoint, args.vocab_size)



    model = get_model(tokenizer, args.model_architecture, args.model_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model = model.to(device)


    dataset = FastaDataset(file_path=args.fasta_path)

    data_collator = GenSLMCollatorForLanguageModeling(
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    train_length = int(np.round(len(dataset) * (1 - args.test_size)))
    lengths = [train_length, len(dataset) - train_length]
    train_dataset, valid_dataset = random_split(dataset, lengths)

    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.weight_decay)
    scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, args.num_train_epochs)

    training_args = get_training_args(
        args.output_dir,
        args.per_device_train_batch_size,
        args.per_device_eval_batch_size,
        args.evaluation_strategy,
        args.eval_steps,
        args.logging_strategy,
        args.logging_steps,
        args.gradient_accumulation_steps,
        args.num_train_epochs,
        args.weight_decay,
        args.warmup_steps,
        args.learning_rate,
        args.save_steps,
        args.fp16,
        args.push_to_hub
    )

    trainer = train_model(model, training_args, tokenizer, train_dataset, valid_dataset, data_collator, device)

    torch.cuda.empty_cache()
    trainer.train()



