'''
NOTE: YOU MUST HAVE PRETRAINED AN UNSUPERVISED MODEL AND HAVE THE WEIGHTS
We currently only support BertForSequenceClassification 
'''

from pathlib import Path
import os
from argparse import ArgumentParser
import torch
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from dataset import SequencePredictionDataset, GenSLMColatorForLanguageModeling, FastaDataset
from utils import read_fasta, parse_sequence_labels, preprocess_data
from dataclasses import dataclass, asdict
import yaml
import wandb

MODEL_DISPATCH = {
    "BertForSequenceClassification": BertForSequenceClassification,
    "bert": BertForSequenceClassification,
}

# this dataclass consolidates the finetuning configuration
@dataclass
class GenSLMFineTuneConfig:
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    gradient_accumulation_steps: int = 2
    model_architecture: str = ""
    model_path: str = ""
    tokenizer_path: str = (
        ""
    )
    output_dir: str = ""
    train_path: str = ""
    validation_path: str = ""
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    logging_strategy: str = "steps"
    logging_steps: int = 500
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    learning_rate: float = 0.00005
    save_steps: int = 500
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    wandb_project: str = ""  # Set to empty string to turn off wandb, otherwise, set as the project name
    fp16: bool = True
    # whether to translate the DNA sequence into protein alphabets
    tokenizer_type: str = ""
    convert_to_aa: bool = True
    num_char_per_token: int = 1  # how many characters per token

    def __post_init__(self):
        # Setting this environment variable enables wandb logging
        if self.wandb_project:
            #os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_PROJECT"] = self.wandb_project
            # Only resume a run if the output path alrimport eady exists
            Path(self.output_dir).mkdir(exist_ok=True, parents=True)
            wandb.init(dir=self.output_dir, resume="auto")
            
            wandb.config.update({"train_config": asdict(self)}, allow_val_change=True)

        # Create the output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Configure tokenization parameters
        if self.tokenizer_type in ["ape_tokenizer", "protein_alphabet_wordlevel"]:
            self.convert_to_aa = True
            self.num_char_per_token = 1
        elif self.tokenizer_type == ["npe_tokenizer", "dna_wordlevel"]:
            self.convert_to_aa = False
            self.num_char_per_token = 1
        elif self.tokenizer_type in ["cpe_tokenizer", "codon_wordlevel"]:
            self.convert_to_aa = False
            self.num_char_per_token = 3
        else:
            raise ValueError(f"Invalid tokenizer_type: {self.tokenizer_type}")

        # Log the config to a yaml file
        with open(os.path.join(self.output_dir, "train_config.yaml"), "w") as fp:
            yaml.dump(asdict(self), fp)

def main():
    """
    example usage in command line format
    python3 seq_pred_finetune.py --config=..finetuning/sample_config.yaml
    """
    
    ID2LABEL={0:'mRNA', 1:'ncRNA', 2:'CDS', 3:'rRNA', 4:'tRNA'}
    LABEL2ID={'mRNA':0, 'ncRNA':1, 'CDS':2, 'rRNA':3, 'tRNA':4}

    # Parse a yaml file to get the training config
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    with open(args.config) as fp:
        config = GenSLMFineTuneConfig(**yaml.safe_load(fp))

    # HuggingFace API training parameters
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
        report_to=["wandb" if config.wandb_project else "none"],
    )

    # Build Tokenizer
    if os.path.isfile(Path(config.tokenizer_path)):
        # These are for the .json files
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=config.tokenizer_path
        )

    else:
        # These are for the bpe tokenizers
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)
        
    model = MODEL_DISPATCH[config.model_architecture].from_pretrained(
            config.model_path, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_dataset = FastaDataset(
        config.train_path,
        num_char_per_token=config.num_char_per_token,
        convert_to_aa=config.convert_to_aa,
        tokenizer_type=config.tokenizer_type,
    )
    eval_dataset = FastaDataset(
        config.validation_path,
        num_char_per_token=config.num_char_per_token,
        convert_to_aa=config.convert_to_aa,
        tokenizer_type=config.tokenizer_type,
    )

    # If the number of tokens in the tokenizer is different from the number of tokens
    # in the model resize the input embedding layer and the MLM prediction head

    # custom DataCollator
    data_collator = GenSLMColatorForLanguageModeling(
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        model_architecture=config.model_architecture
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