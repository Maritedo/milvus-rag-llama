import json
from pathlib import Path
import os

workdir = Path(os.getcwd())

files = [
    workdir / 'data' / 'train_sentence.json',
    workdir / 'data' / 'test_sentence.json',
    # workdir / 'results' / 'bert_base_uncased_llama3.1-70b_10examples.json',
]
datas = []

for file in files:
    with open(file, "r") as f:
        datas.append(json.load(f))

NER = set()
REL = set()
for collection in datas:
    if type(collection) != list:
        collection = list(collection.values())
    for item in collection:
        for entity in item["ner"]:
            NER.add(entity[2])
        for relation in item["relations"]:
            REL.add(relation[4])

print(NER)
print(REL)