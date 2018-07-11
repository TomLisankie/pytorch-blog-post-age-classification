import json
from random import shuffle
from math import ceil

JSON_DIR = "data/blogs/json-data/"
input_file_path = JSON_DIR + "the-data.json"
train_file_path = JSON_DIR + "train.json"
test_file_path = JSON_DIR + "test.json"
train_split = 0.8

print("Loading all data")
with open(input_file_path) as r:
    json_data = json.load(r)
print("Data loaded.")

print("Shuffling the data")
shuffle(json_data)
print("Data shuffled.")

total_data_instances = len(json_data)
train_sample_count = int(ceil(train_split * total_data_instances))
test_sample_count = total_data_instances - train_sample_count

print("Creating a training set of", train_sample_count, "instances.")
train_list = json_data[ : train_sample_count]
print("Training set created.")

print("Creating a testing set of", train_sample_count, "instances.")
test_list = json_data[train_sample_count : ]
print("Testing set created.")

with open(train_file_path, mode='w') as wf:
    print("Saving training set")
    json.dump(train_list, wf, indent=4, sort_keys=False)
    print("Training set saved.")

with open(test_file_path, mode='w') as wf:
    print("Saving testing set")
    json.dump(test_list, wf, indent=4, sort_keys=False)
    print("Testing set saved.")