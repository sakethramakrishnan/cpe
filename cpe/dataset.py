from typing import Any, Dict, List

from torch.utils.data import Dataset

from transformers import BatchEncoding, DataCollatorForLanguageModeling

from utils import (
    any_file_fasta_reader,
    filter_sequences_by_gc,
    group_and_contextualize
)

print('testing git 11/6/23')

class FastaDataset(Dataset):
    def __init__(
        self, file_path: str, num_char_per_token: int, convert_to_aa: bool, tokenizer_type: str
    ) -> None:
        # num_char_per_token is how many characters we tokenize
        # e.g. if our input_seq = 'AATTTGGGAATG' and convert_to_aa == False
        # Say we wanted to tokenize by codons; i.e. ['AAT', 'TTG', 'GGA', 'ATG']
        # then num_char_per_token = 3

        # Read the fasta file
        dna_sequences = any_file_fasta_reader(file_path)
        # Preprocess the sequences into codons
        # TODO: We could also use an <unk> token (this would be better)
        dna_sequences = filter_sequences_by_gc(dna_sequences)

        self.sequences = [
            group_and_contextualize(seq, num_char_per_token, convert_to_aa, tokenizer_type) for seq in dna_sequences
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        # Get the idx'th sequence
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
