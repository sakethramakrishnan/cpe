from torch.utils.data import Dataset
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
class BioDataGenerator(Dataset):
    def __init__(
        self,
        aa_sequences,
        labels,
        tokenizer,
        max_length
        # TODO: Play with filter function abstraction
        # filter_fnxs: Optional[List[Callable[[str], str]]] = None,
    ) -> None:
        # num_char_per_token is how many characters we tokenize
        # e.g. if our input_seq = 'AATTTGGGAATG' and convert_to_aa == False
        # Say we wanted to tokenize by codons; i.e. ['AAT', 'TTG', 'GGA', 'ATG']
        # then num_char_per_token = 3
        

        self.sequences = aa_sequences
        self.labels = labels
        
        assert len(self.sequences) ==  len(self.labels), "There are not the same number of sequences and labels"        
        
        self.tokenizer = tokenizer
        self.tokenized_seqs = tokenizer(self.sequences, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt').to(device)
        self.labels = [torch.tensor(label).to(device) for label in labels]
        
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        # Get the idx'th sequence and label
        tokenized_seq = self.tokenized_seqs[idx]
        return_dict={}
        return_dict['ids'] = torch.tensor(tokenized_seq.ids).to(device)

        return_dict['attention_mask'] = torch.tensor(tokenized_seq.attention_mask).to(device)
        return_dict['token_type_ids'] = torch.tensor(tokenized_seq.type_ids).to(device)
        return return_dict, self.labels[idx].to(device)
    
    
class Optimizer:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 
    
    def zero_grad(self):
        self.optimizer.zero_grad()
