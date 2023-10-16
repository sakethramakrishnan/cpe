from argparse import ArgumentParser
import os
from pathlib import Path

from dataset import FastaDataset, GenSLMCollatorForLanguageModeling

from transformers.trainer_utils import get_last_checkpoint

from transformers import (
    BertForMaskedLM,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    PretrainedConfig
)

import wandb
import yaml
from dataclasses import asdict, dataclass
import json
# TODO: It's good to remove unused imports
from tokenizers import Tokenizer
from os.path import dirname
os.environ["WANDB_DISABLED"] = "true"

MODEL_DISPATCH = {
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "BertForMaskedLM": BertForMaskedLM,
    "neox": GPTNeoXForCausalLM,
    "bert": BertForMaskedLM,
}

# TODO: We can reaplce this logic by checking the file extension (see below)
#       (that removes the need to read the file). File I/O is slow :)
#       This function can be removed after the changes.
def is_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            json.load(file)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False


@dataclass
class GenSLMTrainingConfig:
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    gradient_accumulation_steps: int = 2
    tokenizer_path: str = "/home/couchbucks/Documents/saketh/LLM_sequences/test_tokenizer"
    output_dir: str = "bpe_llm_out"
    train_path: str = "/home/couchbucks/Downloads/all_fasta_files/training/GCA_000977415.2_Sc_YJM1385_v1_genomic_extracted_sequences.fasta"
    validation_path: str = "/home/couchbucks/Downloads/all_fasta_files/training/GCA_000977415.2_Sc_YJM1385_v1_genomic_extracted_sequences.fasta"
    evaluation_strategy: str = 'steps'
    eval_steps: int = 100
    logging_strategy: str = 'steps'
    logging_steps: int = 500
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    # NOTE: in the yaml file and the python file, DO NOT represent lr using scientific notation
    learning_rate: float = 0.00005
    save_steps: int = 500
    load_best_model_at_end: bool = True
    save_total_limit: int = 5
    wandb_project: str = ""  # Set to empty string to turn off wandb
    fp16: bool = True
    convert_to_aa: bool = False  # whether to translate the DNA sequence into protein alphabets
    num_char_per_token: int = 3  # how many characters per token

    def __post_init__(self):

        # Setting this environment variable enables wandb logging
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
            # Only resume a run if the output path already exists
            resume = os.path.exists(self.output_dir)
            Path(self.output_dir).mkdir(exist_ok=True, parents=True)
            wandb.init(dir=self.output_dir, resume=resume)
            wandb.config.update({"train_config": asdict(self)})

        # Create the output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Log the config to a yaml file
        with open(os.path.join(self.output_dir, "train_config.yaml"), "w") as fp:
            yaml.dump(asdict(self), fp)


def main():
    # Parse a yaml file to get the training config
    parser = ArgumentParser()
    # TODO: Instead of a default config file, let's make it a required argument
    #       Then add an example in the README for how to run the training script using this config file 
    #       (we can use a relative path to the config file in the example)
    parser.add_argument("--config", type=str, required=False, default='/home/couchbucks/Documents/saketh/LLM_sequences/cpe/examples/training/train_config.yaml')
    # TODO: Let's add model architecture and model path as fields within the config file
    parser.add_argument("--model_architecture", type=str, required=False, default='bert')
    parser.add_argument("--model_path", type=str, required=False, default='bert/bert_3m.json')

    args = parser.parse_args()
    with open(args.config) as fp:
        config = GenSLMTrainingConfig(**yaml.safe_load(fp))

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        save_steps=config.save_steps,
        fp16=config.fp16,
        load_best_model_at_end=config.load_best_model_at_end,
        save_total_limit=config.save_total_limit,
        push_to_hub=False,
    )

    # Build Tokenizer
    # TODO: Path objects have an is_file() method, it's better to use that.
    #       Although, I don't think we need an if-else check here.
    #       The pretrained_model_name_or_path is the first argument anyways :)
    if os.path.isfile(Path(config.tokenizer_path)):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=config.tokenizer_path)
        
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)

    # Build model
    # TODO: Remove this code: if is_json_file(Path(args.model_path)):
    if Path(args.model_path).suffix == '.json':
        model_config = PretrainedConfig.from_json_file(args.model_path)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.pad_token_id = int(tokenizer.get_vocab()['[PAD]'])
        # TODO: It would be good if we can add these tokens to the tokenizer json file
        #       (that way we don't have to add them here)
        special_tokens = {
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
        tokenizer.add_special_tokens(special_tokens)
        model = MODEL_DISPATCH[args.model_architecture](model_config)
    else:
        # TODO: Why do we need different if-else cases here?
        model = MODEL_DISPATCH[args.model_architecture].from_pretrained(args.model_path)


    # get datasets
    train_dataset = FastaDataset(config.train_path, num_char_per_token=config.num_char_per_token, convert_to_aa=config.convert_to_aa)
    eval_dataset = FastaDataset(config.validation_path, num_char_per_token=config.num_char_per_token, convert_to_aa=config.convert_to_aa)

    # If the number of tokens in the tokenizer is different from the number of tokens
    # in the model resize the input embedding layer and the MLM prediction head

    data_collator = GenSLMCollatorForLanguageModeling(
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # start from checkpoint
    checkpoint = get_last_checkpoint(config.output_dir)
    if checkpoint is not None:
        print("Training from checkpoint:", checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()

