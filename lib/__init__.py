import json
import re
from pathlib import Path
from os import getcwd

workdir = Path(getcwd())

filter_pattern = r"^-[A-Z]{3}-$"

def filter_input(text: str):
    return " ".join([s for s in text.split() if not re.match(filter_pattern, s)])

def require_input(prompt: str, check) -> str:
    text = None
    while not text:
        _text = input(prompt)
        if check(_text):
            text = _text
    return text

def parse_input(raw, relations=False):
    if relations:
        return json.dumps({
            "sentence": filter_input(raw["sentence"]),
            'entities': [[filter_input(entity[0]), entity[1]] for entity in raw["entities"]]
        })
    else:
        return filter_input(raw["sentence"])

def get_sentence_hash(sentence):
    return hash(sentence)  # 生成一个整数哈希值