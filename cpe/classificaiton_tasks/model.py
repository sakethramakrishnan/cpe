import torch.nn as nn
from transformers import BertConfig, BertModel
import torch

class BioBERTModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, num_classes):
        super(BioBERTModel, self).__init__()
        configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads)
        self.transformer = BertModel(configuration)
        
        # additional layers for the classification / regression task
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
            )

    def forward(self, ids, mask, token_type_ids):
        sequence_output, _ = self.transformer(  # the other output is "pooled_output"
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        sequence_output = torch.mean(sequence_output, dim=1)

        result = self.head(sequence_output)
        
        return result
