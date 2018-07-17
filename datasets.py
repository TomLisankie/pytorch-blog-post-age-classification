import torch 
import torch.nn.functional as F
from torch.utils import data
import json

torch.manual_seed(22)

class BlogPostDataset(data.Dataset):

    def __init__(self, data_root_path, json_file_name):
        """
        Args:
            data_root_path (string): directory where all the data files exist
            json_file_name (string): name of the specific JSON file to be represented by this class
        """
        self.data_root_path = data_root_path
        with open(self.data_root_path + json_file_name) as r:
            self.json_data = json.load(r)
        for instance in self.json_data:
            instance["post"] = instance["post"].split(" ")
            instance["gender"] = self.get_gender_as_num(instance["gender"])
            instance["age"] = self.get_age_group(int(instance["age"]))
        
    def get_gender_as_num(self, gender):
        if gender == "male":
            return 0
        else:
            return 1

    def get_age_group(self, age):
        if age < 18:
            # 13 - 17
            return [1, 0, 0]
        elif age < 28:
            # 23 - 27
            return [0, 1, 0]
        elif age < 49:
            # 33 - 48
            return [0, 0, 1]
        else:
            return [0, 0, 0]
    
    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, idx):
        return self.json_data[idx]