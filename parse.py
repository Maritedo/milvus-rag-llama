import json
from pathlib import Path
from os import getcwd
workdir = Path(getcwd())


def get_range(data, index1, index2):
    return " ".join(data['sentence'].split()[index1: index2+1])


def parse(src, suffix="parsed"):
    with open(workdir / 'data' / f'{src}.json', 'r') as f:
        target_datas = json.load(f)

    data2 = [{
        "sentence": data["sentence"],
        "entities": [[get_range(data, ner[0], ner[1]), ner[2]] for ner in data["ner"]],
        "relations": [[get_range(data, rel[0], rel[1]), get_range(data, rel[2], rel[3]), rel[4]] for rel in data["relations"]],
    } for data in target_datas]

    with open(workdir / 'data' / f'{src}.{suffix}.json', 'w') as f:
        f.write(json.dumps(data2).replace("}, {", "},\n{"))


if __name__ == "__main__":
    parse("test_sentence")
    parse("train_sentence")