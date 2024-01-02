import pandas as pd
import os
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, processors, trainers
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Digits, Whitespace
from dataset import BioDataGenerator
from pathlib import Path
import torch
import shutil

TRAIN_DF_NAME = "train.csv"
VALID_DF_NAME = "valid.csv"
TEST_DF_NAME = "test.csv"

def batch_iterator(dataset):
    batch_size = 10000
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def build_bpe_tokenizer(
    corpus_iterator,
    vocab_size,
    tokenizer_type: str,
    add_bos_eos: bool = True,
    max_length: int = 1024,
    save: bool = False,
    initial_alphabet: list[str] = None,
    save_name: str = ''
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
    tokenizer = Tokenizer(BPE(unk_token=special_tokens["unk_token"]))

    if tokenizer_type == "cpe_tokenizer":
        tokenizer.pre_tokenizer = Digits(individual_digits=False)

    else:
        tokenizer.pre_tokenizer = Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        # initial_alphabet=initial_alphabet
    )

    print("Training tokenizer")

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    # Add post-processor
    # trim_offsets=True will ignore spaces, false will leave them in
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
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
        # wrapped_tokenizer.save_pretrained(
        #     f"data/experiment_tokenizers_saved/{tokenizer_type}s/{tokenizer_type}_{tokenizer.get_vocab_size()}"
        # )
        wrapped_tokenizer.save_pretrained(save_name)

    print(f"Returning tokenizer with vocab_size = {tokenizer.get_vocab_size()}")

    return wrapped_tokenizer

def train_biological_tokenizer(data_path, task_type, tokenizer_type, vocab_size, results_path, max_length, logger, tokenizer_path):
    """
    Reads the data from folder, trains the tokenizer, encode the sequences and returns list of data for BERT training
    """
    df_train = pd.read_csv(os.path.join(data_path, TRAIN_DF_NAME))
    df_valid = pd.read_csv(os.path.join(data_path, VALID_DF_NAME))
    df_test = pd.read_csv(os.path.join(data_path, TEST_DF_NAME))

    if task_type in ['REGRESSION', 'regression', 'r']:
        logger.info(f'starting a REGRESSION task!')
        y_train = df_train['label'].astype(float).tolist()
        y_valid = df_valid['label'].astype(float).tolist()
        y_test = df_test['label'].astype(float).tolist()
        
        num_of_classes = 1
    elif task_type in ['TASK_CLASSIFICATION', "CLASSIFICATION", 'classification', 'c']:
        logger.info(f'starting a CLASSIFICATION task!')
        df_train['label_numeric'] = pd.factorize(df_train['label'], sort=True)[0]
        df_valid['label_numeric'] = pd.factorize(df_valid['label'], sort=True)[0]
        df_test['label_numeric'] = pd.factorize(df_test['label'], sort=True)[0]
        y_train = df_train['label_numeric'].astype(int).tolist()
        y_valid = df_valid['label_numeric'].astype(int).tolist()
        y_test = df_test['label_numeric'].astype(int).tolist()
        
        num_of_classes = len(list(set(y_train))) # counts the number different classes
    else:
        exit(f'unknown type of task, got {task_type}. Aviable options are: {TASK_REGRESSION} for regression or {TASK_CLASSIFICATION} for classification')
    
    
    X_train = df_train['seq'].astype(str).tolist()
    X_valid = df_valid['seq'].astype(str).tolist()
    X_test = df_test['seq'].astype(str).tolist()

    if 'WORDS' == tokenizer_type:
        X_train = [' '.join([*aminos]) for aminos in X_train]
        X_valid = [' '.join([*aminos]) for aminos in X_valid]
        X_test = [' '.join([*aminos]) for aminos in X_test]
    elif 'PAIRS' == tokenizer_type:
        def create_pairs(sequences):
            results = []
            for amino in sequences:
                amino_spaces = [*amino]
                if len(amino_spaces[::2]) == len(amino_spaces[1::2]):
                    pairs = [i+j for i,j in zip(amino_spaces[::2], amino_spaces[1::2])]
                elif len(amino_spaces[::2]) < len(amino_spaces[1::2]):
                    lst = amino_spaces[::2].copy()
                    lst.append('')
                    pairs = [i+j for i,j in zip(lst, amino_spaces[1::2])] #add an element to the first list
                else:
                    lst = amino_spaces[1::2].copy()
                    lst.append('')
                    pairs = [i+j for i,j in zip(amino_spaces[::2], lst)] #add an element to the second list
                results.append(' '.join(pairs))
            return results.copy()
        X_train = create_pairs(X_train)
        X_valid = create_pairs(X_valid)
        X_test = create_pairs(X_test)
        
    
    if tokenizer_path not in ['', None]:
    # elif tokenizer_type == 'codon_wordlevel':
    #     tokenizer = PreTrainedTokenizerFast.from_pretrained(
    #         pretrained_model_name_or_path='tokenizer_json_files/codon_wordlevel_71vocab.json'
    #     )
        
    # elif tokenizer_type == 'dna_wordlevel':
    #     tokenizer = PreTrainedTokenizerFast.from_pretrained(
    #         pretrained_model_name_or_path='tokenizer_json_files/dna_wordlevel_100vocab.json'
    #     )
    # elif tokenizer_type == 'protein_alphabet_wordlevel':
    #     tokenizer = PreTrainedTokenizerFast.from_pretrained(
    #         pretrained_model_name_or_path='tokenizer_json_files/protein_alphabet_wordlevel.json'
    #     )
    
        # Build Tokenizer
        if os.path.isfile(Path(tokenizer_path)):
            # These are for the .json files
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
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
        else:
            # These are for the bpe tokenizers
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


    else:
        logger.info(f'starting to train {tokenizer_type} tokenizer...')
        #tokenizer = train_tokenizer(batch_iterator(X_train), tokenizer_type, vocab_size)
        tokenizer = build_bpe_tokenizer(batch_iterator(X_train), vocab_size, 'ape_tokenizer', save=False, save_name='ape_testing_100')
        #if tokenizer_type not in ['BPE', 'bpe', 'bpe_tokenizer', 'ape_tokenizer', 'cpe_tokenizer', 'npe_tokenizer']:
        #    tokenizer.enable_padding(length=max_length)
        logger.info(f'saving tokenizer to {results_path}...')
        #tokenizer.save(os.path.join(results_path, "tokenizer.json"))
    
    def encode(X):
        result = []
        for x in X:
            ids = tokenizer.encode(x).ids
            if len(ids) > max_length:
                ids = ids[:max_length]
            result.append(ids)
        return result

    
    #X_train_ids = tokenizer(X_train, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    #X_valid_ids = tokenizer(X_valid, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    #X_test_ids = tokenizer(X_test, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    #X_train_ids = [torch.tensor(item).to(device) for item in X_train_ids]
    #y_train = [torch.tensor(item).to(device) for item in y_train]
    logger.info('loaded train data to device')
    train_dataset = BioDataGenerator(X_train, y_train, tokenizer, max_length)


    #X_valid_ids = [torch.tensor(item).to(device) for item in X_valid_ids]
    #y_valid = [torch.tensor(item).to(device) for item in y_valid]
    logger.info('loaded valid data to device')
    valid_dataset = BioDataGenerator(X_valid, y_valid, tokenizer, max_length)

    #X_test_ids = [torch.tensor(item).to(device) for item in X_test_ids]
    #y_test = [torch.tensor(item).to(device) for item in y_test]
    logger.info('loaded test data to device')
    test_dataset = BioDataGenerator(X_test, y_test, tokenizer, max_length)
    
    return num_of_classes, train_dataset, valid_dataset, test_dataset


def save_ckp(state, is_best, checkpoint_path, best_model_dir):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)
        
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def load_most_recent_checkpoint(checkpoint_dir, model, optimizer, epoch):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]

    if checkpoint_files:
        # Extract step numbers from checkpoint filenames
        step_numbers = [int(file.split("checkpoint_")[1].split(".pt")[0]) for file in checkpoint_files]

        # Find the index of the checkpoint with the highest step number
        latest_checkpoint_index = step_numbers.index(max(step_numbers))

        # Load the most recent checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[latest_checkpoint_index])
        checkpoint = torch.load(latest_checkpoint_path)
        # Load the model using your preferred method (assuming PyTorch for this example)
        print(checkpoint.keys())

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        print(f"Loaded the most recent checkpoint: {latest_checkpoint_path}")
        return epoch
    else:
        print("No checkpoints found in the specified directory.")