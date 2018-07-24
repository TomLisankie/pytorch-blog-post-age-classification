import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(22)

class BasicLSTMAgeClassifier(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, age_groups_count):
        super(BasicLSTMAgeClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # This LSTM takes word embeddings as inputs and ouputs hidden states with
        # dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.hidden2group = nn.Linear(hidden_dim, age_groups_count)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # Since there is no hidden state yet at the beginning, we start off with a zero-tensor
        # for the state
        # Semantics of the axes are (num_layers, batch_size, hidden_dim)
        num_layers = 1
        batch_size = 1
        return (torch.zeros(num_layers, batch_size, self.hidden_dim),
                torch.zeros(num_layers, batch_size, self.hidden_dim))
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        group_space = self.hidden2group(lstm_out.view(len(sentence), -1))
        group_scores = F.log_softmax(group_space, dim = 1)
        print(type(group_scores))
        print("Group Scores Length:", len(group_scores))
        return group_scores