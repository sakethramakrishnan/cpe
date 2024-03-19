#Testing
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

from dataset import FastaDataset, GenSLMColatorForLanguageModeling, FastaDatasetTokenized
import evaluate

from utils import cpe_decode, get_aligned_seqs, get_num_correct, get_num_correct_punish, build_bpe_tokenizer

import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

MODEL_DISPATCH = {
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "BertForMaskedLM": BertForMaskedLM,
    "neox": GPTNeoXForCausalLM,
    "bert": BertForMaskedLM,
    "GPT": GPTNeoXForCausalLM,
    "gpt": GPTNeoXForCausalLM
}

BPE_TOKENIZERS = ['ape_tokenizer', 'bpe_tokenizer', 'cpe_tokenizer', 'npe_tokenizer']

# this dataclass consolidates the training configuration
@dataclass
class GenSLMTrainingConfig:
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    gradient_accumulation_steps: int = 2
    model_architecture: str = ""
    model_path: str = ""
    max_length: int = 1080

    tokenizer_path: str = ""
    mlm: bool = True
    mlm_probability: float = 0.15
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
    vocab_size: int = 0

    convert_to_aa: bool = True
    num_char_per_token: int = 1  # how many characters per token

    def __post_init__(self):
        # Setting this environment variable enables wandb logging
        if self.wandb_project:
            #os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_PROJECT"] = self.wandb_project
            # Only resume a run if the output path alrimport eady exists
            resume = os.path.exists(self.output_dir)
            Path(self.output_dir).mkdir(exist_ok=True, parents=True)
            wandb.init(dir=self.output_dir, resume="auto")
            
            wandb.config.update({"train_config": asdict(self)}, allow_val_change=True)
        # Create the output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # Configure tokenization parameters
        if self.tokenizer_type in ["ape_tokenizer", "protein_alphabet_wordlevel"]:
            self.convert_to_aa = True
            self.num_char_per_token = 1
        elif self.tokenizer_type in ["npe_tokenizer", "dna_wordlevel"]:
            self.convert_to_aa = False
            self.num_char_per_token = 1
        elif self.tokenizer_type in ["cpe_tokenizer", "codon_wordlevel"]:
            self.convert_to_aa = False
            self.num_char_per_token = 3
        else:
            raise ValueError(f"Invalid tokenizer_type: {self.tokenizer_type}")

        # for some reason, the default mlm_probability is a tuple
        if type(self.mlm_probability) == tuple:
            self.mlm_probability = float(self.mlm_probability[0])
        
        # Log the config to a yaml file
        with open(os.path.join(self.output_dir, "train_config.yaml"), "w") as fp:
            yaml.dump(asdict(self), fp)


def main():
    """
    example usage in command line format
    python3 main_llm.py --config=../examples/training/train_config.yaml
    """

    # Parse a yaml file to get the training config
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    with open(args.config) as fp:
        config = GenSLMTrainingConfig(**yaml.safe_load(fp))

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



    # get datasets
    if config.tokenizer_type == 'dna_wordlevel' and config.model_architecture in ['bert', 'Bert', 'BERT'] and 1 == 0:
        train_dataset = FastaDatasetTokenized(
            config.train_path,
            num_char_per_token=config.num_char_per_token,
            convert_to_aa=config.convert_to_aa,
            tokenizer_type=config.tokenizer_type,
            tokenizer=tokenizer
        )
        eval_dataset = FastaDatasetTokenized(
            config.validation_path,
            num_char_per_token=config.num_char_per_token,
            convert_to_aa=config.convert_to_aa,
            tokenizer_type=config.tokenizer_type,
            tokenizer=tokenizer
        )

    else:
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

    print(f"{len(train_dataset)} training samples.")
    print(f"{len(eval_dataset)} evaluation samples.")


    # Build Tokenizer
    if os.path.isfile(Path(config.tokenizer_path)):
        # These are for the .json files
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=config.tokenizer_path
        )

    else:
        # These are for the bpe tokenizers

        if config.tokenizer_path not in ["", None]:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)
        else:
            
            # build bpe tokenizer on the fly

            full_dataset = train_dataset + eval_dataset
            tokenizer = build_bpe_tokenizer(
                    corpus_iterator = full_dataset,
                    vocab_size = config.vocab_size,
                    tokenizer_type = config.tokenizer_type,
                    save = False
                    )



    # Build model

    # if we are instantiating a new model, we need to instantiate a new model from a json file (if statement)
    if Path(config.model_path).suffix == ".json":
        model_config = PretrainedConfig.from_json_file(config.model_path)



        # the following are only for gpt models
        if config.model_architecture in ['neox', 'NeoX', "GPT", 'gpt']:
            model_config.use_parallel_residual = True
            model_config.attention_dropout = 0.0
            model_config.hidden_dropout = 0.0
            model_config.rope_scaling = None

        
        special_tokens = {
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }

        # for some reason, we need to add the special tokens even though they are in the json file

        tokenizer.add_special_tokens(special_tokens)

        model_config.pad_token_id = int(tokenizer.vocab["[PAD]"])
        model_config.vocab_size = len(tokenizer.vocab)

        model = MODEL_DISPATCH[config.model_architecture](model_config)
    else:
        # There are different if-else cases because:
        # a) if we have an untrained model, we need to instantiate a new model from a json file (if statement)
        # b) if we have a trained model, we need to load it from a checkpoint (else statement)
        model = MODEL_DISPATCH[config.model_architecture].from_pretrained(
            config.model_path
        )
        
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Training model with name `{config.model_architecture}` "
        f"- Total size={n_params/2**20:.2f}M params or {n_params}"
    )

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred  
        
        if config.model_architecture == 'bert' or config.mlm:
            predictions = np.argmax(predictions, axis=2) #axis is 2 for just MLM this works for bert
        else:
            predictions = np.argmax(predictions[0], axis=2) # idk if this is how you do it for gpt style
            
        predictions_list = []
        labels_list = []
        
        total_correct_count = 0
        total_total_count = 0
        
        for pred, acc in zip(predictions, labels):
            
            if config.mlm:
                # because there are masked tokens and non masked tokens, we need to decode MLM separately
                print_str = ' '.join(['NaN' if token == -100 else tokenizer.decode(token) for token in acc]) # NaN represents the sequences that don't matter
                #print_str = tokenizer.decode(acc).replace("[", "")

                #print(f"Acc gtruth: {print_str}") # Keep the label like this so that we can more easily compare label and predictions
                
                prediction_str = tokenizer.decode(pred).replace("[", "")
                #print(pred)
                #print(f"Prediction: {prediction_str.replace(']', '')}") # replace the brackets with empty spaces to make it easier to read


                prediction_list = []
                label_list = []


                # we do the following to only extract the labels that the model predicted
                for i, token in enumerate(acc):
                        if token != -100:
                            prediction_list.append(pred[i])
                            label_list.append(token)


                #print(f"Predicted Tokens (need not be in order in samples): {tokenizer.decode(prediction_list)}")
                #print(f"Actual Tokens (need not be in order in samples): {tokenizer.decode(label_list)}")

                assert len(prediction_list) == len(label_list), "the predictions and labels are not the same length for MASKED LM"
                
                predictions_list.extend(prediction_list)
                labels_list.extend(label_list)     
                

            else:
                            
                # we need to do the following 2 lines bc for GPT models, the labels have the BOS token, 
                # while the predictions don't have the BOS token; so we just truncate the BOS token from the labels
                # and we just remove all the "excess" tokens form the predictions to match the lengths of the labels and predictions
                
                
                if config.model_architecture in ['neox', 'NeoX', "GPT", 'gpt']:
                    label_list = acc[1:]  
                    label_list = [token for token in label_list if token != -100] # -100 is all of the pad tokens  
                    prediction_list = pred[:len(label_list)]
                else:
                    # this is for the case for non-masked BERT model
                    prediction_list = pred
                    label_list = [x if x != -100 else int(tokenizer.vocab["[PAD]"]) for x in acc][1:]  # the -100 should only be for MLM and during training for the softmax (-100 will be ignored)
                    prediction_list = pred[:len(label_list)]
                    # take out the BOS token
                
                if config.tokenizer_type == "cpe_tokenizer":
                    
                    predicted_seq = cpe_decode(tokenizer.decode(prediction_list).replace('[EOS]', ''))
                    actual_seq = cpe_decode(tokenizer.decode(label_list).replace('[EOS]', ''))
                    
                else:
                    predicted_seq = (tokenizer.decode(prediction_list)).replace(" ", "")
                    actual_seq = (tokenizer.decode(label_list)).replace(" ", "")

                #print(f"Prediction: {predicted_seq}")
                #print(f"Acc gtruth: {actual_seq}")

                
                # if config.tokenizer_type in BPE_TOKENIZERS:
                ga_align1, ga_align2, ga_alignment_score, la_align1, la_align2, la_alignment_score = get_aligned_seqs(predicted_seq, actual_seq)
                
                assert len(ga_align1) == len(ga_align2), "Global alignment not same length"
                assert len(la_align1) == len(la_align2), "Local alignment not same length"
                
                if ga_alignment_score > la_alignment_score:
                    best_pred = ga_align1
                    best_actual = ga_align2
                    
                else:
                    best_pred = la_align1
                    best_actual = la_align2
                # else:
                #     best_pred, best_actual = pad_nucleotides(predicted_seq, actual_seq)
                    
                
                # print(f"Prediction: {best_pred}")
                # print(f"Acc gtruth: {best_actual}")
                
                correct_count, total_count = get_num_correct(best_pred, best_actual)
                total_correct_count += correct_count
                total_total_count += total_count
                
                
        
        if total_correct_count != 0:
            accuracy = total_correct_count / total_total_count if total_total_count > 0 else 0
            accuracy_dict = {}
            accuracy_dict['accuracy'] = accuracy  
        else:
            accuracy_dict = metric.compute(predictions=predictions_list, references=labels_list)
        
        
        #print(accuracy_dict)

        return accuracy_dict


    # If the number of tokens in the tokenizer is different from the number of tokens
    # in the model resize the input embedding layer and the MLM prediction head

    # if config.tokenizer_type == 'dna_wordlevel' and config.model_architecture in ['bert', 'Bert', 'BERT']:
    #     print('inside')
    #     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

  
    # custom DataCollator
    data_collator = GenSLMColatorForLanguageModeling(
        train_mode=True,
        tokenizer=tokenizer,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability, # for some reason the mlm_probability is stored as a tuple
        model_architecture=config.model_architecture,
        max_length = config.max_length
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #compute_metrics=compute_metrics,
    )

    # start from checkpoint
    checkpoint = get_last_checkpoint(config.output_dir)
    if checkpoint is not None:
        print("Training from checkpoint:", checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
