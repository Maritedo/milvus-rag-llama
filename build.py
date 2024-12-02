from lib.collections import *
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
    insert_sentence(sentences, ner_data, relations_data)
    collection.load()  # 加载数据到内存
    query_result = collection.query(
        expr="id >= 0",  # 条件表达式，检索所有数据
        output_fields=["sentence", "ner", "relations"],
    )
    print("\nQuery Results:")
    for item in query_result:
        print("Sentence:", item["sentence"])
        print("NER:", json.loads(item["ner"]))
        print("Relations:", json.loads(item["relations"]))