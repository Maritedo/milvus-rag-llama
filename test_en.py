import requests
from lib.cache import QueryCache
from lib.utils import LocalEmbbeder, ServerEmbedder
from lib import parse_input
import os
from pathlib import Path
import json
import time
import traceback

workdir = Path(os.getcwd())

with open(workdir / "data" / "test_sentence.new.json", "r") as f:
    json_data = json.load(f)
    test_sentences = json_data[:250]
with open(workdir / "data" / "train_sentence.new.json", "r") as f:
    json_data = json.load(f)
    train_sentences = json_data

schema_ner = """All entity types are as below:
1.Task: Applications, problems to solve, systems to construct.
2.Method: Methods , models, systems to use, tools/components of a system/frameworks.
3.Metric: Metrics, measures, or entities that can express quality of a system/method.
4.Material: Data, datasets, resources, Corpus, Knowledge base.
5.OtherScientificTerms: Phrases that are a scientific terms but do not fall into any of the above classes.
6.Generic: General terms or pronouns that may refer to a entity but are not themselves informative, often used as connection words."""
schema_rel = """All relation types are as below:
1.USED-FOR: B is used for A, B models A, A is trained on B, B exploits A, A is based on B.
2.FEATURE-OF: B belongs to A, B is a feature of A, B is under A domain.
3.HYPONYM-OF: B is a hyponym of A, B is a type of A.
4.PART-OF: B is a part of A.
5.COMPARE: Symmetric relation. Opposite of conjunction, compare two models/methods, or listing two opposing entities.
6.CONJUNCTION: Symmetric relation. Function as similar role or use/incorporate with."""

def get_completion(sentence, examples, model):
    prompt = f"""As a NLP api, you will receive JSON input that includes a sentence and its indexed slices, then you need to extract entities and relations of entities from inputed sentence. Make sure that the result you give MUST be pure JSON string. DO NOT give codes or explanation.
{schema_ner}
{schema_rel}
Parse the structure of the sentence and identify which type of entity each grammatical structure belongs to. After that find out all relations between these entities and classify the relations. Expected output is a JSON string that represent a list, each item in list has 2 fields(both are list, not dict): "ner" and "relations". Items in 'ner' are tuples as [entity, start_index, end_index, entity_type] (entity is string, start_index and end_index corespond to slice indexes in the input, entity_type must be Task, Method, Metric, Material, OtherScientificTerms or Generic). Items in 'relations' are tuples like [start_index_of_A, end_index_of_A, start_index_of_B, end_index_of_B, relation_type]. Notice that entity types and relation types should be one of the types listed above. For examples:
{"\n".join([f'Input {i+1}: {parse_input(train_sentences[item[0]]["sentence"])}\nOutput {i+1}: {{"ner": {json.dumps(train_sentences[item[0]]["ner"])}, "relations": {json.dumps(train_sentences[item[0]]["relations"])}}}' for i, item in enumerate(examples)])}
Now wait for a input and give the output."""
    api_url = "http://172.16.129.30:11434/api/generate"
    payload = {
        "model": model,
        "system": prompt,
        "prompt": parse_input(sentence),
        "stream": False,
        # "max_tokens": 1024,  # 设置生成文本的最大长度
        # "temperature": 0.7, # 调节生成文本的多样性
        # "top_p": 0.9        # 调节生成文本的随机性
    }
    response = requests.post(api_url, json=payload)
    return response.json()

from lib.evaluate import count_contained_intervals


def check_output(output):
    if "ner" not in output.keys() or len(output["ner"]) == 0 or type(output["ner"][0]) != list:
        raise Exception(f"Invalid output skeleton (no ner field): {output}")
    else:
        len_ner = set(len(i) for i in output["ner"])
        if len(len_ner) > 0 and len_ner != {4}:
            raise Exception(f"Invalid output structure (bad ner field): {output}")
    if "relations" not in output.keys() or (len(output["relations"]) != 0 and type(output["relations"][0]) != list):
        raise Exception(f"Invalid output skeleton (no relations field): {output}")
    else:
        len_rel = set(len(i) for i in output["relations"])
        if len(len_rel) > 0 and len_rel != {5}:
            raise Exception(f"Invalid output structure (bad relations field): {output}")
    return True

fixed_exps = [(557,), (1515,), (1295,), (1028,), (697,), (376,), (273,), (247,), (242,), (1779,), (1365,)]

def evaluate(results_file, results, model, embedder, numofexps, fixed, quite, title):
    from lib.dynamic import DynamicDisplay, DynamicText
    logs = []
    logs_scroll = []
    total = [0, 0, 0] # sum_test sum_gen matched
    for index in range(len(results)):
        generated = results[str(index)]
        target = test_sentences[index]
        re2 = count_contained_intervals(target["ner"], generated["ner"])
        total[0] += len(target["ner"])
        total[1] += len(generated["ner"])
        total[2] += re2[0]
    with DynamicDisplay(5, extra=[logs_scroll], extra_height=16) as display:
        def log(msgs, flag=-1): # -1 means logging, otherwise print to logs list but do not print to console and setline
            if flag == -1:
                for msg in msgs.split("\n"):
                    logs_scroll.append(msg)
                    logs.append(msg)
            else:
                display.set_line(flag, msgs)
                for msg in msgs.split("\n"):
                    logs.append(msg)
        display.set_line(0, title)
        status_line = DynamicText(lambda: f"[Entity] Expected:{total[0]} Generated:{total[1]} Matched:{total[2]}")
        display.set_line(1, status_line)
        statistics_line = DynamicText(lambda: f"[Status] Recall: {(total[2] / (total[0]) if total[0] else 0) * 100:.2f}%, Precision: {(total[2] / total[1] if total[0] else 0) * 100:.2f}%")
        display.set_line(2, statistics_line)
        _interrupted = False
        if not fixed:
            cache = QueryCache(embedder.name(), db_path=workdir / "cache" / "cache.db")
        else:
            exps = fixed_exps
        next = 0
        checkpointfile = (
            workdir / "record" / (f'evaluate_{embedder.name()}_{model.replace(":", "-")}_{numofexps}exps{"_fixed" if fixed else ""}.next')
        )
        if os.path.exists(checkpointfile):
            with open(checkpointfile, "r+") as f:
                fc = f.read()
                if fc and fc.strip().isdigit():
                    next = int(fc)
        if next >= len(test_sentences):
            return False
        log(f"Starting from index {next} ({next} finished)")
        batch_size = 1 # int(input("Enter batch size: "))
        try:
            while True:
                if next >= len(test_sentences):
                    break
                end = min(next + batch_size, len(test_sentences))
                log(f"Processing index {next}{f" to {end - 1}" if end-1==next else ""}...", flag=3)
                current_time = time.time()
                for index in range(next, end):
                    if not fixed:
                        exps = cache.fetch_results(next)
                    while True:
                        step = 0
                        try:
                            step += 1  # 1
                            res = get_completion(test_sentences[index]["sentence"], exps[:numofexps], model)
                            step += 1  # 2
                            output: dict = json.loads(res["response"])
                            step += 1  # 3
                            check_output(output)
                            step += 1  # 4
                            results[str(index)] = output
                            total[0] += len(test_sentences[index]["ner"])
                            total[1] += len(output["ner"])
                            total[2] += count_contained_intervals(test_sentences[index]["ner"], output["ner"])[0]
                            break
                        except KeyboardInterrupt:
                            if step == 4:
                                results[str(index)] = output
                                total[0] += len(test_sentences[index]["ner"])
                                total[1] += len(output["ner"])
                                total[2] += count_contained_intervals(test_sentences[index]["ner"], output["ner"])[0]
                            raise KeyboardInterrupt
                        except Exception as e:
                            if not quite:
                                log(traceback.format_exc())
                                if step >= 2: log(f"Output: {res['response']}")
                            log("Error occurred, retrying...")
                            continue
                log(f"Time elapsed: {time.time() - current_time:.2f}s, Length: {len(output['ner'])} + {len(output['relations'])}")
                matched, recall, precision = count_contained_intervals(test_sentences[index]["ner"], output["ner"])
                log(f"[{index}] Recall: {recall * 100:.2f}%, Precision: {precision * 100:.2f}%, Matched: {matched}/{len(test_sentences[index]['ner'])}")
                next = min(next + batch_size, len(test_sentences))
                with open(checkpointfile, "w") as f:
                    f.write(str(next))
        except KeyboardInterrupt:
            log("\nExiting...")
            _next = max(next, index)
            _interrupted = True
            with open(checkpointfile, "w") as f:
                f.write(str(_next))
        finally:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            from datetime import datetime
            with open(workdir / "logs" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{embedder.name()}_{model.replace(':', '-')}_{numofexps}exps{"_fixed" if fixed else ""}.log", "w") as f:
                f.write("\n".join(logs))
        return _interrupted

tasks = [
    {
        "model": "llama3.1:70b",
        "numofexps": 3,
        "quite": False,
        "fixed": False,
    },{
        "model": "llama3.1:70b",
        "numofexps": 3,
        "quite": False,
        "fixed": True,
    },{
        "model": "llama3.1:70b",
        "numofexps": 1,
        "quite": False,
        "fixed": False,
    },{
        "model": "llama3.1:70b",
        "numofexps": 1,
        "quite": False,
        "fixed": True,
    },{
        "model": "llama3.2:3b",
        "numofexps": 3,
        "quite": False,
        "fixed": False,
    },{
        "model": "llama3.2:3b",
        "numofexps": 3,
        "quite": False,
        "fixed": True,
    },{
        "model": "llama3.2:3b",
        "numofexps": 1,
        "quite": False,
        "fixed": False,
    },{
        "model": "llama3.2:3b",
        "numofexps": 1,
        "quite": False,
        "fixed": True,
    }
]


def main():
    if not os.path.exists(workdir / "results"):
        os.mkdir(workdir / "results")
    if not os.path.exists(workdir / "record"):
        os.mkdir(workdir / "record")
    if not os.path.exists(workdir / "logs"):
        os.mkdir(workdir / "logs")
    for index, task in enumerate(tasks):
        model = task["model"]
        numofexps = task["numofexps"]
        quite = task["quite"]
        fixed = task["fixed"]
        embedder = ServerEmbedder("http://172.16.129.30:11434", model_name=model)
        title = f"Task {index+1}:: Embedder={embedder.name()} Model={model.replace(':', '-')} Examples={numofexps}{',Fixed' if fixed else ''}"
        # embedder = LocalEmbbeder(
            # model_name="google-bert/bert-base-uncased"
            # model_name='sentence-transformers/all-MiniLM-L6-v2'
        # )
        results_file = (workdir / "results" / f"{embedder.name()}_{model.replace(":", "-")}_{numofexps}exps{"_fixed" if fixed else ""}.json")
        results: dict = None
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                fc = f.read()
                results = json.loads(fc if fc else "{}")
        else:
            results = {}
        _interrupted = evaluate(results_file, results, model, embedder, numofexps, fixed, quite, title=title)
        if _interrupted:
            break

if __name__ == "__main__":
    main()