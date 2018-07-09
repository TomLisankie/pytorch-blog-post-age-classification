# Inspired by https://github.com/mihaibogdan10/json-reuters-21578/blob/master/convert_sgm_to_json.py 

from bs4 import BeautifulSoup
import os
import json
from os.path import join, isfile, dirname

# Needs to pour all posts into a single file where each post has an age and gender attached to it.

xmlDataDirPath = join(dirname(__file__), 'data/blogs/xml-data')
jsonDataDirPath = join(dirname(__file__), 'data/blogs/json-data')

''' Returns a single data sample to be appended to the JSON file'''
def get_single_sample(post, gender, age):
    return {'post': post, 'gender': gender, 'age': age}

files = [f for f in os.listdir(xmlDataDirPath) if isfile(join(xmlDataDirPath, f))]
files_length = len(files)
counter = 0
jsonName = "the-data.json"
processed_count = 0
json_docs = []
for xmlName in files:
    print("Starting on", xmlName)
    fileNameElements = xmlName.split('.') # So we can get the gender and age of the person

    # Load what's already in the JSON file into memory
    try:
        with open(join(jsonDataDirPath, jsonName)) as r_json:
            current_json_data = json.load(r_json)
            if current_json_data is None:
                current_json_data = []
    except IOError:
        print("Could not read file, starting from scratch")
        current_json_data = []

    with open(join(xmlDataDirPath, xmlName), "rb") as rf:
        content = BeautifulSoup(rf, features = "xml")

        new_data = []
        for entry in content.findAll('Blog'):
            posts = [elem.text.replace("\n", "").replace("\t", "").strip() for elem in entry.findAll('post')]
            for post in posts:
                new_data.append(get_single_sample(post, fileNameElements[1], fileNameElements[2]))
        
        for instance in new_data:
            json_docs.append(instance)
        
        print("Done with", xmlName)
        processed_count += 1
        print("Done with", processed_count, "of", files_length)

with open(join(jsonDataDirPath, jsonName), mode='w') as wf:
    json.dump(json_docs, wf, indent=4, sort_keys=False)