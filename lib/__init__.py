def require_input(prompt: str, check) -> str:
    text = None
    while not text:
        _text = input(prompt)
        if check(_text):
            text = _text
    return text

import json
def parse_input(t):
    return str(json.dumps({
        'sentence': t,
        'slices': [(i,e) for i,e in enumerate(t.split())]
    }))