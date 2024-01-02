from utils import  train_biological_tokenizer
from argparse import ArgumentParser
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import os
import torch
from model import BioBERTModel
from torch.optim import AdamW
from dataset import Optimizer
from utils import save_ckp, load_ckp, load_most_recent_checkpoint
import torch.nn as nn
import time
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy import stats

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())



TASK_REGRESSION = "REGRESSION"
TASK_CLASSIFICATION = "CLASSIFICATION"

TOKEZNIER_BPE = "BPE"
TOKEZNIER_WPC = "WPC"
TOKEZNIER_UNI = "UNI"
TOKEZNIER_WORDS = "WORDS"
TOKEZNIER_PAIRS = "PAIRS"

UNK_TOKEN = "<UNK>"  # token for unknown words
SPL_TOKENS = [UNK_TOKEN]  # special tokens

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# this dataclass consolidates the training configuration
@dataclass
class TrainingConfig:
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    per_device_test_batch_size: int = 64
    num_workers: int = 4
    bpe_vocab_size: int = 200
    gradient_accumulation_steps: int = 2
    model_architecture: str = ""
    model_path: str = ""
    num_layers: int = 2
    num_attention_heads: int = 2
    hidden_size: int = 128
    mlm: bool = True
    mlm_probability: float = 0.15,
    output_dir: str = ""
    data_path: str = ""
    task_type: str = ''
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    logging_strategy: str = "steps"
    logging_steps: int = 500
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    learning_rate: float = 0.00005
    save_steps: int = 500
    max_length: int = 512
    save_total_limit: int = 1
    wandb_project: str = ""  # Set to empty string to turn off wandb, otherwise, set as the project name
    run_name: str = ''
    fp16: bool = True
    tokenizer_path: str = ''
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
            resume = os.path.exists(self.output_dir)
            Path(self.output_dir).mkdir(exist_ok=True, parents=True)

            wandb.init(dir=self.output_dir, resume="auto", name=self.run_name)

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

        
def train_model(model, task_type, train_generator, valid_generator, test_generator, epochs, results_path, start_epoch, config, optimizer):
    if task_type in ['REGRESSION', 'regression', 'r']:
        loss_fn = nn.MSELoss()
    elif task_type in ['TASK_CLASSIFICATION', "CLASSIFICATION", 'classification', 'c']:
        loss_fn = nn.CrossEntropyLoss()
    
    
    def calc_metrics_regression(model, generator):
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,y in generator:
                outputs = model(ids=x['ids'], mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
                # print(outputs)
                outputs = outputs.to(torch.float)
                y_pred.extend(outputs.tolist())
                y = y.to(torch.float)
                y_true.extend(y.tolist())
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            loss = loss_fn(y_pred, y_true).item()    
            #loss = loss / len(generator)
            spearman = stats.spearmanr(y_pred, y_true)
        return loss, spearman[0], spearman[1]
    
    def calc_metrics_classification(model, generator):
        loss = 0
        output_list = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,y in generator:
                outputs = model(ids=x['ids'], mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
                output_list.extend(outputs.tolist())
                y_pred.extend(torch.argmax(outputs, dim=1).int().tolist())
                y_true.extend(y.int().tolist())
                
            y_pred = torch.tensor(y_pred)
            y_true = torch.tensor(y_true)
            output_tensor = torch.tensor(output_list)

            loss = loss_fn(output_tensor, y_true).item()   
            mcc = matthews_corrcoef(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
        return loss, mcc, accuracy
        
    list_of_rows = []
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f'----- starting epoch = {start_epoch} -----')
        epoch_loss = 0.0
        running_loss = 0.0
        # Training
        start_time = time.time()
        model.train()
        for idx, (x, y) in enumerate(train_generator):
            
            
            outputs = model(ids=x['ids'], mask=x['attention_mask'], token_type_ids=x['token_type_ids'])
            outputs = outputs.squeeze(dim=1) # otherwise, it would be (batch_size, 1), when we only want it to be (batch_size) to compare with labels
            loss = loss_fn(outputs, y)
            loss.backward()
            if ((idx + 1) % config.gradient_accumulation_steps == 0) or (idx + 1 == len(train_generator)):
                optimizer.step()
                optimizer.zero_grad()
                
            # print statistics
            running_loss += loss.item()
            if idx % config.logging_steps == 0:
                end_time = time.time()
                loss = running_loss / config.logging_steps
                logger.info('[%d, %5d] time: %.3f loss: %.3f' % (epoch, idx + 1, end_time - start_time, loss))
                running_loss = 0.0
                if config.wandb_project:
                    wandb.log({"training_loss": loss})
                start_time = time.time()

            if idx % config.eval_steps == 0:
                #print('inside')
                model.eval()
                if task_type in ['REGRESSION', 'regression', 'r']:
                    val_loss, spearman_val_corr, spearman_val_p = calc_metrics_regression(model, valid_generator)
                    test_loss, spearman_test_corr, spearman_test_p = calc_metrics_regression(model, test_generator)

                    logger.info(f'epoch = {epoch}, val_loss = {val_loss}, spearman_val_corr = {spearman_val_corr}, spearman_val_p = {spearman_val_p}, test_loss = {test_loss}, spearman_test_corr = {spearman_test_corr}, spearman_test_p = {spearman_test_p}')
                    list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'spearman_val_corr': spearman_val_corr, 'spearman_val_p': spearman_val_p, 'test_loss': test_loss, 'spearman_test_corr': spearman_test_corr, 'spearman_test_p': spearman_test_p})
                    if config.wandb_project:
                        wandb.log({'val_loss': val_loss, 'spearman_val_corr': spearman_val_corr, 'spearman_val_p': spearman_val_p, 'test_loss': test_loss, 'spearman_test_corr': spearman_test_corr, 'spearman_test_p': spearman_test_p})
                elif task_type in ['TASK_CLASSIFICATION', "CLASSIFICATION", 'classification', 'c']:
                    val_loss, val_mcc, accuracy_val = calc_metrics_classification(model, valid_generator)
                    test_loss, test_mcc, accuracy_test = calc_metrics_classification(model, test_generator)

                    logger.info(f'epoch = {epoch}, val_loss = {val_loss}, val_mcc = {val_mcc}, test_loss = {test_loss}, test_mcc = {test_mcc}, accuracy_val = {accuracy_val}, accuracy_test = {accuracy_test}')
                    list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'val_mcc': val_mcc, 'test_loss': test_loss, 'test_mcc': test_mcc, 'accuracy_val': accuracy_val,'accuracy_test': accuracy_test})
                    if config.wandb_project:
                        wandb.log({'val_loss': val_loss, 'val_mcc': val_mcc, 'test_loss': test_loss, 'test_mcc': test_mcc, 'accuracy_val': accuracy_val, 'accuracy_test': accuracy_test})
        
            if idx % config.save_steps == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                is_best = True # TODO: find the last MCC and see if its better now
                save_ckp(checkpoint, is_best, checkpoint_path=os.path.join(config.output_dir, f"checkpoint_{epoch}.pt"), best_model_dir=os.path.join(config.output_dir, "best_model_dir"))
 
    df_loss = pd.DataFrame(list_of_rows)
    df_loss.to_csv(os.path.join(results_path, f"results.csv"), index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    with open(args.config) as fp:
        config = TrainingConfig(**yaml.safe_load(fp))
        
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.output_dir not in [None, ""]:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'best_model_dir'), exist_ok=True)
    
    num_classes, train_dataset, valid_dataset, test_dataset = train_biological_tokenizer(
        config.data_path, 
        config.task_type, 
        config.tokenizer_type, 
        config.bpe_vocab_size, 
        config.output_dir, 
        config.max_length, 
        logger,
        config.tokenizer_path
        )
    
    model = BioBERTModel(config.hidden_size, config.num_layers, config.num_attention_heads, num_classes)

    model.to(device)
    
    logger.info('loaded model to device')
    logger.info(f'device is {device}')
    
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Training model with name `BertModel` "
        f"- Total size={n_params/2**20:.2f}M params or {n_params}"
    )
    logger.info(f'num of paramters = {n_params}')
    
    adam = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    optimizer = Optimizer(n_params, config.warmup_steps, adam)
    start_epoch = 0
    
    if config.model_path not in [None, '']:
        model, optimizer, start_epoch = load_ckp(cpk_path=config.model_path, model=model, optimizer=optimizer)
    elif config.output_dir not in [None, '']:
        start_epoch = load_most_recent_checkpoint(config.output_dir, model, optimizer, start_epoch)
        print(start_epoch)
    
    g = torch.Generator()
    g.manual_seed(0)
 
    train_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=config.num_workers, batch_size=config.per_device_train_batch_size, generator=g)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, shuffle=True, num_workers=config.num_workers, batch_size=config.per_device_eval_batch_size, generator=g)
    test_generator = torch.utils.data.DataLoader(test_dataset, shuffle=True, num_workers=config.num_workers, batch_size=config.per_device_test_batch_size, generator=g)
    train_model(
        model, 
        config.task_type, 
        train_generator, 
        valid_generator, 
        test_generator, 
        config.num_train_epochs, 
        config.output_dir, 
        start_epoch, 
        config, 
        optimizer
        ) # TODO: why so many config. ?


if __name__ == "__main__":
    main()
    
    