def require_input(prompt: str, check) -> str:
    text = None
    while not text:
        _text = input(prompt)
        if check(_text):
            text = _text
    return text

def parse_input(t):
    return str({
        'sentence': t,
        'slices': [(i,e) for i,e in enumerate(t.split())]
    })