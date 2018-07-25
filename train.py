import torch
import torch.nn as nn
import torch.optim as optim
import json
import models
import datasets

torch.manual_seed(22)

blog_posts_data_dir = "data/blogs/json-data/"
train_file_name = "train.json"
test_file_name = "test.json"

training_set = datasets.BlogPostDataset(blog_posts_data_dir, train_file_name)

# Map each word to a unique int value
word_to_int = {}
words = []
for instance in training_set:
    for word in instance["post"]:
        if word not in word_to_int:
            word_to_int[word] = len(word_to_int)
        if word not in words:
            words.append(word)

def prepare_sequence(seq, word_to_int):
    ints = [word_to_int[w] for w in seq]
    return torch.tensor(ints, dtype = torch.long)

# Train the model
EMBEDDING_DIM = 32
HIDDEN_DIM = 15
NUM_AGE_GROUPS = 3
model = models.BasicLSTMAgeClassifier(EMBEDDING_DIM, words, HIDDEN_DIM, NUM_AGE_GROUPS)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# See what the scores are before training
with torch.no_grad():
    inputs = prepare_sequence(training_set[1]["post"], word_to_int)
    group_scores = model(inputs)
    print(group_scores)

for epoch in range(300):
    for instance in training_set[ : 4000]:
        
        # Zero-out the gradients
        model.zero_grad()

        # Zero-out hidden state from previous instance
        model.hidden = model.init_hidden()

        sentence_in = prepare_sequence(instance["post"], word_to_int)
        group = instance["age"]
        print("Group Length:", len(group))

        group_scores = model(sentence_in)

        loss = loss_function(group_scores, torch.tensor(group, dtype = torch.long))
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_set[1]["post"], word_to_int)
    group_scores = model(inputs)
    print(group_scores)


# TODO: Save the model