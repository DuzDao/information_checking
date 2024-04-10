import torch
import torch.nn as nn
import torch.nn.functional as F

class PickEvidenceIndexModel(nn.Module):
    def __init__(self, config):
        super(PickEvidenceIndexModel, self).__init__()
        self.linear1 = nn.Linear(config["embedding"]["embedding_dim"]*2, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, config["data"]["num_labels"])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config["model"]["dropout"])

    def forward(self, input_ids):
        input = self.dropout(input_ids)
        input = self.linear1(input)
        input = self.relu(input)
        input = self.linear2(input)
        input = self.relu(input)
        input = self.linear3(input)
        out = self.relu(input)
        out_soft = F.softmax(out)
        return out, out_soft # = logits, out_softmax