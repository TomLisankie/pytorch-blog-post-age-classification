import torch
import torch.nn as nn
import torch.nn.functional as F
import bcolz
import pickle

torch.manual_seed(22)

class BasicLSTMAgeClassifier(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, age_groups_count):
        super(BasicLSTMAgeClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # Create GloVe dictionary
        glove_path = "data/embeddings/glove.6B"
        vectors = bcolz.open(f"{glove_path}/6B.50.dat")[:]
        words = pickle.load(open(f"{glove_path}/6B.50_words.pkl", "rb"))
        word2idx = pickle.load(open(f"{glove_path}/6B.50_idx.pkl", "rb"))
        glove = {w : vectors[word2idx[w]] for w in words}

        # Create Embedding layer from the loaded GloVe embeddings
        matrix_len = len(vocab)
        weights_matrix = np.zeros((matrix_len, 50))
        words_found = 0
        for i, word in enumerate(vocab):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale = 0.6, size = (embedding_dim))
        print(words_found, "out of", len(vocab), "found in GloVe embeddings")
        num_embeddings, embedding_dim = weights_matrix.size()
        self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.word_embeddings.load_state_dict({"weight" : weights_matrix})
        self.word_embeddings.weight.requires_grad = True # Trainable for now
        
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