import json
from pathlib import Path

with open(Path(__file__).parent / 'data' / 'train_sentence.json', "r") as f:
    json_data = json.load(f)
    train_sentences = json_data
    
with open(Path(__file__).parent / 'data' / 'test_sentence.json', "r") as f:
    json_data = json.load(f)
    test_sentences = json_data

datas = [train_sentences, test_sentences]
NER = set()
REL = set()
for collection in datas:
    for item in collection:
        for entity in item["ner"]:
            NER.add(entity[2])
        for relation in item["relations"]:
            REL.add(relation[4])

print(NER)
print(REL)