import json

with open("data/train_sentence.json", "r") as f:
    target_datas = json.load(f)

data2 = [{
    "sentence": data["sentence"],
    "ner": [[" ".join(data["sentence"].split()[ner[0]:ner[1]+1]), ner[0], ner[1], ner[2]] for ner in data["ner"]],
    "relations": data["relations"],
} for data in target_datas]

json.dump(data2, open("data/train_sentence.new.json", "w"))