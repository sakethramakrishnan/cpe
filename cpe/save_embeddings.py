
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import torch
import numpy.typing as npt

from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerFast, BertForMaskedLM

from dataset import FastaDataset

from argparse import ArgumentParser

CODON_TO_CHAR = {
    "TCG": "A",
    "GCA": "B",
    "CTT": "C",
    "ATT": "D",
    "TTA": "E",
    "GGG": "F",
    "CGT": "G",
    "TAA": "H",
    "AAA": "I",
    "CTC": "J",
    "AGT": "K",
    "CCA": "L",
    "TGT": "M",
    "GCC": "N",
    "GTT": "O",
    "ATA": "P",
    "TAC": "Q",
    "TTT": "R",
    "TGC": "S",
    "CAC": "T",
    "ACG": "U",
    "CCC": "V",
    "ATC": "W",
    "CAT": "X",
    "AGA": "Y",
    "GAG": "Z",
    "GTG": "a",
    "GGT": "b",
    "GCT": "c",
    "TTC": "d",
    "AAC": "e",
    "TAT": "f",
    "GTA": "g",
    "CCG": "h",
    "ACA": "i",
    "CGA": "j",
    "TAG": "k",
    "CTG": "l",
    "GGA": "m",
    "ATG": "n",
    "TCT": "o",
    "CGG": "p",
    "GAT": "q",
    "ACC": "r",
    "GAC": "s",
    "GTC": "t",
    "TGG": "u",
    "CCT": "v",
    "GAA": "w",
    "TCA": "x",
    "CAA": "y",
    "AAT": "z",
    "ACT": "0",
    "GCG": "1",
    "GGC": "2",
    "CTA": "3",
    "AAG": "4",
    "AGG": "5",
    "CAG": "6",
    "AGC": "7",
    "CGC": "8",
    "TTG": "9",
    "TCC": "!",
    "TGA": "@",
    "XXX": "*",
}

parser = ArgumentParser()
parser.add_argument("--fasta_path", type=str, required=True)

parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--save_name", type=str, required=True)
args = parser.parse_args()

# enter the fasta filepath to a fasta path:
fasta_path = args.fasta_path

# enter the checkpoint to the tokenizer:
tokenizer_path = args.tokenizer_path


model_checkpoint = args.model_checkpoint

# ImportError: cannot import name 'GenSLMColatorForLanguageModeling' from 'dataset' (/home/couchbucks/Documents/saketh/cpe/cpe/dataset.py)
from transformers import BatchEncoding, DataCollatorForLanguageModeling
class GenSLMColatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Augment the underlying DataCollatorForLanguageModeling to handle
    multiple batch encoding inputs."""

    def __init__(self, train_mode: bool = False, **kwargs) -> None:
        self.train_mode = train_mode
        super().__init__(**kwargs)

    def tokenize(self, sequences: List[str]) -> BatchEncoding:
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_special_tokens_mask=self.train_mode and self.mlm,
            max_length=1024,
        )

    def torch_call(self, examples: List[str]) -> Dict[str, Any]:
        # First, tokenize the batch
        batch = self.tokenize(examples)

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

from dataset import FastaDataset

def generate_embeddings_and_logits(model, dataloader):
    embeddings, logits, input_ids = [], [], []
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(model.device)
            outputs = model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            seq_lengths = batch.attention_mask.sum(axis=1)
            for seq_len, hidden, logit, input_id in zip(
                seq_lengths, last_hidden_states, outputs.logits, batch.input_ids
            ):
                # Get averaged embedding
                embedding = hidden[1 : seq_len - 1, :].mean(dim=0).cpu().numpy()
                embeddings.append(embedding)
                # Get logits
                # TODO: Determine if the lsoftmax should be calculated before or after the splice
                logits.append(lsoftmax(logit[1 : seq_len - 1, :]).cpu().numpy())
                # Get input_ids
                input_ids.append(input_id[1 : seq_len - 1].cpu().numpy())

    return np.array(embeddings), logits, input_ids



def llm_inference(
    tokenizer_path: Path,
    model_path: Path,
    fasta_path: Path,
    return_codon: bool,
    return_aminoacid: bool,
    batch_size: int,
    fasta_contains_aminoacid: bool = False,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    
    
    if os.path.isfile(Path(tokenizer_path)):
        # These are for the .json files
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path
        )

    else:
        # These are for the bpe tokenizers
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
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

    model = BertForMaskedLM.from_pretrained(model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if fasta_contains_aminoacid:
        dataset = FastaAminoAcidDataset(file_path=fasta_path)
    else:
        dataset = FastaDataset(
            file_path=fasta_path,
            num_char_per_token = 3,
            convert_to_aa = False,
            tokenizer_type = "cpe_tokenizer"
        )

    data_collator = GenSLMColatorForLanguageModeling(
        train_mode=False,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    embeddings, logits, input_ids = generate_embeddings_and_logits(model, dataloader)

    return embeddings, logits, input_ids

embeddings, _, _ = llm_inference(
    tokenizer_path,
    model_checkpoint,
    fasta_path,
    return_codon = False,
    return_aminoacid = False,
    batch_size = 128,
    fasta_contains_aminoacid = False,
)

np.save("bert_3m_cpe_tokenizer_mdh_embeddings.py", embeddings)