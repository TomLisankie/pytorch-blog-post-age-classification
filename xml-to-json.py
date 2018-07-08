# Inspired by https://github.com/mihaibogdan10/json-reuters-21578/blob/master/convert_sgm_to_json.py 

from bs4 import BeautifulSoup
import os
import json
from os.path import join, isfile, dirname
from unidecode import unidecode

xmlDataDirPath = join(dirname(__file__), 'data/blogs/xml-data')
jsonDataDirPath = join(dirname(__file__), 'data/blogs/json-data')

def xml_node_to_json(node, gender, age):
    # Needs to get all dates and posts from each blog

    return {
        'posts': [elem.text.replace("\n", "").replace("\t", "").strip() for elem in node.findAll('post')],
        'gender': gender,
        'age': age
    }


files = [f for f in os.listdir(xmlDataDirPath) if isfile(join(xmlDataDirPath, f))]

for xmlName in files:
    print("Starting on", xmlName)
    jsonName = xmlName.replace('.xml', '.json')
    fileNameElements = xmlName.split('.')
    with open(join(xmlDataDirPath, xmlName), "rb") as rf:
        with open(join(jsonDataDirPath, jsonName), mode='w') as wf:
            content = BeautifulSoup(rf, features = "xml")

            jsonDocs = []
            for entry in content.findAll('Blog'):
                data = xml_node_to_json(entry, fileNameElements[1], fileNameElements[2])
                jsonDocs.append(data)

            json.dump(jsonDocs, wf, indent=4, sort_keys=False)
            print("Done with", xmlName)