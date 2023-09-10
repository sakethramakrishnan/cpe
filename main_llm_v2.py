import argparse

from typing import Optional

from pathlib import Path
import bpe_tokenizer
import evaluate

import os
import numpy as np

import torch

from transformers import (
    BertForMaskedLM,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    GPTNeoXForCausalLM,
)

from torch.utils.data import random_split

from dataset import FastaDataset, GenSLMCollatorForLanguageModeling

from utils import get_model_no_path

os.environ["WANDB_DISABLED"] = "true"
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

def get_model(tokenizer, model_checkpoint: Optional = None, model_json_path: Optional = None, model_architecture: Optional = None):
    if model_checkpoint:
        model = BertForMaskedLM.from_pretrained(Path(model_checkpoint))

    elif model_json_path:
        if "bert" in model_json_path:
            config = PretrainedConfig.from_json_file(model_json_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = BertForMaskedLM(config)
        elif "neox" in model_json_path:
            config = PretrainedConfig.from_json_file(model_json_path)
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            model = GPTNeoXForCausalLM(config)
        else:
            raise ValueError('Your model is not a bert or neox model')

    elif model_architecture:
        model = get_model_no_path(tokenizer, model_architecture)

    else:
        raise ValueError(
            'Please provide either: '
            'a) a valid model checkpoint in the "model_checkpoint" argument'
            'b) a path to a json file with the model configuration in the "model_json_path" argument'
            'c) valid model architecture in the "model_architecture" argument'
        )

    return model


if __name__ == "__main__":

    args = get_args()

    # TODO: Assume tokenizer already exists
    #sequences = get_sequences(args.fasta_path)
    tokenizer = get_tokenizer(args.tokenizer_checkpoint, args.vocab_size)

    model = get_model(tokenizer, args.model_architecture, args.model_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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


    training_args = TrainingArguments(
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

    # Build trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # TODO: Save a checkpoint at the very end
    trainer.train()
