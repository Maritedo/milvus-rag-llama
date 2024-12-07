import json

def require_input(prompt: str, check) -> str:
    text = None
    while not text:
        _text = input(prompt)
        if check(_text):
            text = _text
    return text

def parse_input(t):
    return str(json.dumps({
        'sentence': t,
        'slices': [(i,e) for i,e in enumerate(t.split())]
    }))
    

def get_sentence_hash(sentence):
    return hash(sentence)  # 生成一个整数哈希值