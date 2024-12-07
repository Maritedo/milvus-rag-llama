import requests
from lib.cache import QueryCache
from lib.utility import embedder
from lib import parse_input
import os
from pathlib import Path
import json
import time

workdir = Path(os.getcwd())


with open(workdir / 'data' / 'test_sentence.json', "r") as f:
    json_data = json.load(f)
    test_sentences = json_data
with open(workdir / 'data' / 'train_sentence.json', "r") as f:
    json_data = json.load(f)
    train_sentences = json_data

schema_ner = \
"""All possible entity types are as below:
1.Task: Applications, problems to solve, systems to construct.
2.Method: Methods , models, systems to use, or tools, components of a system,frameworks.
3.Metric: Metrics, measures, or entities that can express quality of a system/method.
4.Material: Data, datasets, resources, Corpus, Knowledge base.
5.OtherScientiﬁcTerms: Phrases that are a scientiﬁc terms but do not fall into any of the above classes.
6.Generic: General terms or pronouns that may refer to a entity but are not themselves informative, often used as connection words."""
schema_rel = \
"""All possible relationship types and their meanings are as below:
1.USED-FOR: B is used for A, B models A, A is trained on B, B exploits A, A is based on B.
2.FEATURE-OF: B belongs to A, B is a feature of A, B is under A domain.
3.HYPONYM-OF: B is a hyponym of A, B is a type of A.
4.PART-OF: B is a part of A.
5.COMPARE: Symmetric relation (use blue to denote entity). Opposite of conjunction, compare two models/methods, or listing two opposing entities.
6.CONJUNCTION: Symmetric relation (use blue to denote entity). Function as similar role or use/incorporate with.
"""

def get_completion(sentence, examples):
    prompt = \
f"""As a natural language processing api, you should extract the Named Entities from given sliced sentence (a JSON list) and provide their type in a JSON format. Make sure entities you find are meaningful and the result you give MUST be pure JSON string and contains NO extra information. DO NOT give codes or explanation.
{schema_ner}
{schema_rel}
Example of an input and its corresponding output is as below:
{"\n".join([f'Input {i+1}: {parse_input(train_sentences[item[0]]["sentence"])}\nOutput {i+1}: {{"ner": {json.dumps(train_sentences[item[0]]["ner"])}, "relations": {json.dumps(train_sentences[item[0]]["relations"])}}}' for i, item in enumerate(examples)])}
Numbers in output that appear in pairs are starting index and stoping index of entities(!IMPORTAMT! Indexes start with 0).
Wait for my inputs and give reasonable outputs according to the instruction above."""
    api_url = "http://172.16.129.30:11434/api/generate"
    payload = {
        "model": "llama3.1:70b",  # 替换为你使用的具体模型名称
        "system": prompt,
        "prompt": parse_input(sentence),
        "stream": False,
        # "max_tokens": 200,  # 设置生成文本的最大长度
        # "temperature": 0.7,  # 调节生成文本的多样性
        # "top_p": 0.9        # 调节生成文本的随机性
    }
    response = requests.post(api_url, json=payload)
    return response.json()

numofexps = 10
if not os.path.exists(workdir / 'results'):
    os.mkdir(workdir / 'results')
results_file = workdir / 'results' / f"{embedder.name()}_{numofexps}examples.json"
with open(results_file, "r+") as f:
    fc = f.read()
    results: dict = json.loads(fc if fc else "{}")

if __name__ == "__main__":
    cache = QueryCache(embedder.name(), db_path=workdir / 'cache' / 'cache.db')
    next = 0
    checkpointfile = workdir / 'record' / ('evaluate_' + embedder.name() + '_llama3_1.next')
    with open(checkpointfile, 'r+') as f:
        fc = f.read()
        if fc and fc.strip().isdigit():
            next = int(fc)
    print(f"Starting from index {next} ({next} finished)")
    batch_size = int(input("Enter batch size: "))
    try:
        while True:
            if next >= len(test_sentences):
                break
            current_time = time.time()
            end = min(next + batch_size, len(test_sentences))
            print(f"Processing index {next}{f" to {end - 1}" if end-1==next else ""}...")
            items = test_sentences[next:next+batch_size]
            for index in range(next, end):
                exps = cache.fetch_results(next)
                while True:
                    step = 0
                    try:
                        step += 1 # 1
                        res = get_completion(test_sentences[index]["sentence"], exps)
                        step += 1 # 2
                        output = json.loads(res["response"].replace("'", '"'))
                        step += 1 # 3
                        results[index] = output
                        step += 1 # 4
                        break
                    except KeyboardInterrupt:
                        if step == 3:
                            results[index] = output
                        raise KeyboardInterrupt
                    except:
                        print("Error occurred, retrying...")
                        continue
            next = min(next + batch_size, len(test_sentences))
            with open(checkpointfile, 'w') as f:
                f.write(str(next))
            print(f"Time elapsed: {time.time() - current_time:.2f}s")
    except KeyboardInterrupt:
        print("\nExiting...")
        _next = max(next, index)
        with open(checkpointfile, 'w') as f:
            f.write(str(_next))
    finally:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)