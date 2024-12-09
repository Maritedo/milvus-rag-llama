import json
import re
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

def parse_input(t):
    return str(json.dumps({
        'slices': {i:e for i,e in enumerate(t.split())},
        'sentence': t
    }))
    

def get_sentence_hash(sentence):
    return hash(sentence)  # 生成一个整数哈希值