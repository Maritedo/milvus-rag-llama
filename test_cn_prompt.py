from datetime import datetime
import requests
from lib.cache import QueryCache
from lib.utility import LocalEmbbeder, ServerEmbedder
from lib import parse_input, filter_input, workdir
import os
import json
import time
import traceback

with open(workdir / "data" / "test_sentence.parsed.json", "r") as f:
    json_data = json.load(f)
    test_sentences = json_data[:250]
with open(workdir / "data" / "train_sentence.parsed.json", "r") as f:
    json_data = json.load(f)
    train_sentences = json_data

schema_ner = """所有可能的实体类型:
1.Task: 任务、目标。如要实现的应用、要解决的问题、要构建的系统
2.Method: 使用的方法、模型、系统、工具等，以及系统或框架的组件
3.Metric: 能够表达系统或方法的质量的指标、度量、实体
4.Material: 数据（集）、资源、知识库、语料库
5.OtherScientificTerms: 属于属于科学术语、但不属于上述任何类别的短语
6.Generic: 一般术语或指代一个实体的代词，本身并不提供信息，通常用作连接词"""
schema_rel = """所有可能的关系类型:
1.USED-FOR: B用于A、B对A进行建模、A基于B训练、B利用A、A基于B.
2.FEATURE-OF: B属于A、B描述A、B在A的范围内.
3.HYPONYM-OF: B是A的下位词，B是A的一种类型
4.PART-OF: B是A的一部分
5.COMPARE: 对称关系。与CONJUNCTION相对，比较两种模型或方法，或列举两个对立实体。
6.CONJUNCTION: 对称关系，A与B作用相似、一同使用或互相协同
7.EVALUATE-FOR: A验证B、A评估B、A核实B"""


def extract_entity_types(entities):
    entity_dict = {}
    for entity, entity_type in entities:
        if entity_type not in entity_dict:
            entity_dict[entity_type] = []
        entity_dict[entity_type].append(entity)
    return entity_dict

def remove_mark(output):
    filtered = {
        "entities": [[filter_input(ent[0]), ent[1]] for ent in output['entities']],
        "relations": [[filter_input(rel[0]), filter_input(rel[1]), rel[2]] for rel in output['relations']]
    }
    return json.dumps(filtered)

def get_completion(sentence, examples, model):
    # entities_exp = extract_entity_types([item for exp_index in examples for item in train_sentences[exp_index[0]]["entities"]])
    prompt = f"""你是一个命名实体抽取专家，你需要从输入的句子中提取尽可能多的实体，并找出提取出的实体之间的关系（若存在）。不要给出代码或解释说明。
{schema_ner}
{schema_rel}
一些输入和应给出的输出示例如下：
{"\n".join([f'Input {i+1}: {filter_input(train_sentences[item[0]]["sentence"])}\nOutput {i+1}: {remove_mark(train_sentences[item[0]])}' for i, item in enumerate(examples)])}
在提取实体以及找出提取出的实体的关系时，应提取尽可能多的结果，并且不应对内容进行改写。等待输入并给出结果..."""
    # with open(workdir / "tmp" / f"prompt_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as f:
    #     f.write(prompt)
    api_url = "http://172.16.129.30:11434/api/generate"
    payload = {
        "model": model,
        "system": prompt,
        "prompt": parse_input(sentence),
        "stream": False,
        "format": "json",
        # "max_tokens": 2048,  # 设置生成文本的最大长度
        # "temperature": 0.9, # 调节生成文本的多样性
        # "top_p": 0.9        # 调节生成文本的随机性
    }
    response = requests.post(api_url, json=payload)
    return response.json()

"""
并且给出部分实体类型的例子：
{"\n".join([f"{k}: {', '.join(v[:3])}" for k, v in entities_exp.items()])}"""

from lib.evaluate import count_contained_intervals


def check_output(output):
    if "entities" not in output.keys() or len(output["entities"]) == 0 or type(output["entities"][0]) != list:
        raise Exception(f"Invalid output skeleton (no entities field): {output}")
    else:
        len_ner = set(len(i) for i in output["entities"])
        if len(len_ner) > 0 and len_ner != {2}:
            raise Exception(f"Invalid output structure (bad entities field): {output}")
    if "relations" not in output.keys() or (len(output["relations"]) != 0 and type(output["relations"][0]) != list):
        raise Exception(f"Invalid output skeleton (no relations field): {output}")
    else:
        len_rel = set(len(i) for i in output["relations"])
        if len(len_rel) > 0 and len_rel != {3}:
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
        re2 = count_contained_intervals(target["entities"], generated["entities"])
        total[0] += len(target["entities"])
        total[1] += len(generated["entities"])
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
                            sentence = filter_input(test_sentences[index]["sentence"])
                            log(f"Input: {sentence}")
                            res = get_completion(test_sentences[index]["sentence"], exps[:numofexps], model)
                            step += 1  # 2
                            output: dict = json.loads(res["response"])
                            step += 1  # 3
                            check_output(output)
                            step += 1  # 4
                            results[str(index)] = output
                            total[0] += len(test_sentences[index]["entities"])
                            total[1] += len(output["entities"])
                            total[2] += count_contained_intervals(test_sentences[index]["entities"], output["entities"])[0]
                            log(f"Output: {output}")
                            break
                        except KeyboardInterrupt:
                            if step == 4:
                                results[str(index)] = output
                                total[0] += len(test_sentences[index]["entities"])
                                total[1] += len(output["entities"])
                                total[2] += count_contained_intervals(test_sentences[index]["entities"], output["entities"])[0]
                                log(f"Output: {output}")
                            raise KeyboardInterrupt
                        except Exception as e:
                            if not quite:
                                log(traceback.format_exc())
                                if step >= 2: log(f"Output: {res['response']}")
                            log("Error occurred, retrying...")
                            continue
                log(f"Time elapsed: {time.time() - current_time:.2f}s, Length: {len(output['entities'])} + {len(output['relations'])}")
                matched, recall, precision = count_contained_intervals(test_sentences[index]["entities"], output["entities"])
                log(f"[{index}] Recall: {recall * 100:.2f}%, Precision: {precision * 100:.2f}%, Matched: {matched}/{len(test_sentences[index]['entities'])}")
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