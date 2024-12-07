import os
import time
from lib.utility import *
from pathlib import Path
import json

workdir = Path(__file__).resolve().parent

# 示例句子、NER 和关系数据
with open(workdir / "data" / "train_sentence.json", "r") as f:
    json_data = json.load(f)
    sentences = [item["sentence"] for item in json_data]
    ner_data = [item["ner"] for item in json_data]
    relations_data = [item["relations"] for item in json_data]

if __name__ == "__main__":
    if not os.path.exists("record"):
        os.makedirs("record")
    next = 0
    with open(workdir / 'record' / (embedder.name()+'.next'), 'r+') as f:
        fc = f.read()
        if fc and fc.strip().isdigit():
            next = int(fc)
    print(f"Starting from index {next} ({next} finished)")
    batch_size = int(input("Enter batch size: "))
    while True:
        if next >= len(sentences):
            break
        current_time = time.time()
        print(f"Processing {next} to {min(next + batch_size, len(sentences))}...")
        insert_sentence(sentences[next:next+batch_size], ner_data[next:next+batch_size], relations_data[next:next+batch_size])
        next = min(next + batch_size, len(sentences))
        with open(workdir / 'record' / (embedder.name() + '.next'), 'w') as f:
            f.write(str(next))
        print(f"Time elapsed: {time.time() - current_time:.2f}s")