from typing import Any, Dict, List, Tuple
import numpy.typing as npt
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BatchEncoding, DataCollatorForLanguageModeling
from utils import any_file_fasta_reader, group_and_contextualize
from transformers import BertForMaskedLM, GPTNeoXForCausalLM, PreTrainedTokenizerFast
import os
from torch.utils.data import DataLoader

# from typing import Callable, Optional
MODEL_DISPATCH = {
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "BertForMaskedLM": BertForMaskedLM,
    "neox": GPTNeoXForCausalLM,
    "bert": BertForMaskedLM,
}

class FastaDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        num_char_per_token: int,
        convert_to_aa: bool,
        tokenizer_type: str,
        # TODO: Play with filter function abstraction
        # filter_fnxs: Optional[List[Callable[[str], str]]] = None,
    ) -> None:
        # num_char_per_token is how many characters we tokenize
        # e.g. if our input_seq = 'AATTTGGGAATG' and convert_to_aa == False
        # Say we wanted to tokenize by codons; i.e. ['AAT', 'TTG', 'GGA', 'ATG']
        # then num_char_per_token = 3

        # Read the fasta file
        dna_sequences = any_file_fasta_reader(file_path)
        # Preprocess the sequences into codons
        # TODO: We could also use an <unk> token (this would be better)
        

        self.sequences = [
            group_and_contextualize(
                seq, num_char_per_token, convert_to_aa, tokenizer_type
            )
            for seq in dna_sequences
        ]
        print(len(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        # Get the idx'th sequence
        return self.sequences[idx]


class GenSLMColatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Augment the underlying DataCollatorForLanguageModeling to handle
    multiple batch encoding inputs."""

    def __init__(self, model_architecture, train_mode: bool = False, **kwargs) -> None:
        self.train_mode = train_mode
        self.model_architecture = model_architecture
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
        
        if self.model_architecture in ['neox', "NeoX", "GPT", 'gpt']:    
            if 'token_type_ids' in batch:
                batch.pop('token_type_ids')
            
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
    model_type: str,
    model_path: Path,
    fasta_path: Path,
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

    model = MODEL_DISPATCH[model_type].from_pretrained(model_path)
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
