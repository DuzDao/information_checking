import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class TextEmbedding(nn.Module):
    def __init__(self, config):
        super(TextEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config["embedding"]["pretrained_name"])
        self.embedding_type = config["embedding"]["embedding_type"]

        if self.embedding_type == 1:
            self.embedding = SentenceTransformer("keepitreal/vietnamese-sbert")
        else:
            self.embedding = AutoModel.from_pretrained(config["embedding"]["pretrained_name"])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.add_special_tokens = config["tokenizer"]["add_special_tokens"]
        self.padding = config["tokenizer"]["padding"]
        self.truncation = config["tokenizer"]["truncation"]
        self.max_length = config["tokenizer"]["max_length"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"]
        self.return_tensors = config["tokenizer"]["return_tensors"]
    
    def get_token_ids(self, text):
        token_ids = self.tokenizer.batch_encode_plus( 
            text,
            add_special_tokens = self.add_special_tokens,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length,
            return_attention_mask = self.return_attention_mask,
            return_tensors = self.return_tensors
        )
        return token_ids

    def forward(self, claim, context):
        if self.embedding_type == 1: #incase using sentence-transformers
            claim_embedding = torch.tensor(self.embedding.encode(claim, show_progress_bar=False)).to(self.device)
            context_embedding = torch.tensor(self.embedding.encode(context, show_progress_bar=False)).to(self.device)
        else:
            claim_token_ids = self.get_token_ids(claim)
            evidence_token_ids = self.get_token_ids(context)

            claim_embedding = self.embedding(claim_token_ids.input_ids.to(self.device), claim_token_ids.attention_mask.to(self.device))
            evidence_embedding = self.embedding(evidence_token_ids.input_ids.to(self.device), evidence_token_ids.attention_mask.to(self.device))

            claim_embedding = claim_embedding.last_hidden_state[:, 0, :]
            evidence_embedding = evidence_embedding.last_hidden_state[:, 0, :]
        
        input_ids = torch.cat((claim_embedding, context_embedding), dim = 1)

        return input_ids
