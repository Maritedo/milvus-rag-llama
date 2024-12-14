import requests
import os
import json
import time
import traceback

from lib import parse_input, filter_input, workdir
from lib.cache import QueryCache
from lib.embedder import LocalEmbedder, ServerEmbedder
from lib.utils import KeyboardInterruptTemporaryIgnored, load_checkpoint, get_time_str, check_element_layout
from lib.evaluate import count_contained_intervals


with open(workdir / "data" / "test_sentence.parsed.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    test_sentences = json_data
with open(workdir / "data" / "train_sentence.parsed.json", "r", encoding="utf-8") as f:
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

def remove_mark(element):
    filtered = {
        "sentence": filter_input(element['sentence']),
        "entities": [[filter_input(ent[0]), ent[1]] for ent in element['entities']]
    }
    return json.dumps(filtered)

def get_completion(element, examples, model):
    # entities_exp = extract_entity_types([item for exp_index in examples for item in train_sentences[exp_index[0]]["entities"]])
    system = f"""你是一个SPO三元组抽取专家，我将输入一个句子和句子的实体提取结果，你需要根据语境，找出给定实体的关系。不要给出代码或解释说明。
{schema_ner}
{schema_rel}
一些输入和应给出的输出示例如下（format: {{"relations": [[entityA, entityB, rtype], ...]}}）：
{"\n".join([f'Input {i+1}: {remove_mark(train_sentences[item[0]])}\nOutput {i+1}: {{"relations": {[[filter_input(rel[0]), filter_input(rel[1]), rel[2]] for rel in train_sentences[item[0]]['relations']]}}}' for i, item in enumerate(examples)])}
在提取实体的关系时，应提取尽可能多的结果，并且不应对内容进行改写。等待输入并给出结果..."""
    prompt = parse_input(element, True)
    api_url = "http://172.16.129.30:11434/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        # "max_tokens": 2048,  # 设置生成文本的最大长度
        # "temperature": 0.9, # 调节生成文本的多样性
        # "top_p": 0.9        # 调节生成文本的随机性
    }
    if __debug__:
        with open(workdir / 'prompt' / f"prompt_{get_time_str()}.txt", "w", encoding="utf-8") as f:
            f.write(system)
            f.write("\n\n")
            f.write(prompt)
    response = requests.post(api_url, json=payload)
    return response.json()


def check_output(output):
    if type(output) != dict:
        raise Exception(f"Invalid output type: {type(output)}")
    if "relations" not in output.keys() or type(output["relations"]) != list:
        raise Exception(f"Invalid output skeleton (no valid relations field): {output}")
    else:
        if len(output["relations"]) != 0 and set([type(rel) for rel in output["relations"]]) != {list}:
            raise Exception(f"Invalid output structure (bad type of 'relations'): {output}")
        for rel in output["relations"]:
            if not check_element_layout(rel, str, str, str):
                raise Exception(f"Invalid output structure (bad type of elements in 'relations'): {output}")
            
    return True

fixed_exps = [(557,), (1515,), (1295,), (1028,), (697,), (376,), (273,), (247,), (242,), (1779,), (1365,)]

def evaluation_worker(index, model, exps, logger, max_retries=10, quite=False) -> tuple[bool, bool, dict]:
    """
    @return: (successed, interrupted, output)
    """
    retries = 0
    output = None
    while True:
        try:
            logger(f"Input: {parse_input(test_sentences[index])}")
            res = get_completion(test_sentences[index], exps, model)
            with KeyboardInterruptTemporaryIgnored():
                output = json.loads(res["response"])
                check_output(output)
                return (True, False, output)
        except KeyboardInterrupt:
            return (False, True, None)
        except Exception as e:
            quite or logger(traceback.format_exc())
            logger("Error occurred, retrying...")
            if retries >= max_retries:
                logger("Retries exceeded, skipping...")
                return (False, False, None)
            retries += 1
            continue
            

def evaluate(results_file, results, model, embedder, title, identifier=None, numofexps=0, fixed=False, quite=False, batch_size=1):
    from lib.dynamic import DynamicDisplay, DynamicText
    logs = []
    logs_scroll = []
    total = [0, 0, 0, 0] # sum_test sum_gen matched fails
    for key in results.keys():
        generated = results[str(key)]
        target = test_sentences[int(key)]
        total[0] += len(target["relations"])
        total[1] += len(generated["relations"])
        total[2] += count_contained_intervals(target["relations"], generated["relations"])[0]
    
    if identifier is None:
        identifier = f"{get_time_str()}"
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
        status_line = DynamicText(lambda: f"[Relati] Expected:{total[0]} Generated:{total[1]} Matched:{total[2]}")
        display.set_line(1, status_line)
        statis_line = DynamicText(lambda: f"[Status] Recall: {(total[2] / (total[0]) if total[0] else 0) * 100:.2f}%, Precision: {(total[2] / total[1] if total[1] else 0) * 100:.2f}%, F1 Score: {(2 * total[2] / (total[0] + total[1]) if total[0] + total[1] else 0):.4f}")
        display.set_line(2, statis_line)
        extras_info = DynamicText(lambda: f"[Extras] Failure: {total[3]}")
        display.set_line(3, extras_info)
        # initialize examples configuration
        exps = None
        if not fixed:
            cache = QueryCache(embedder.name(), db_path=workdir / "cache" / "cache.db")
        else:
            exps = fixed_exps
        # try to resume from last checkpoint    
        checkpointfile = workdir / "record" / (f'evaluate_{identifier}.next')
        next = int(load_checkpoint(checkpointfile, '0'))
        # calculate failure count
        total[3] = next - len(results)
        # check if the evaluation has done
        if next >= len(test_sentences):
            return False
        # main loop
        _last_successed = False
        log(f"Starting from index {next} ({next} finished)")
        try:
            while True:
                if next >= len(test_sentences):
                    return False
                end = min(next + batch_size, len(test_sentences))
                log(f"Processing index {next}{f" to {end - 1}" if end-1!=next else ""}...", flag=4)
                
                for next in range(next, end):
                    current_time = time.time()
                    _last_successed = False
                    if not fixed:
                        exps = cache.fetch_results(next)
                    _last_successed, _interrupted, output = evaluation_worker(
                        next,
                        model,
                        exps[:numofexps],
                        logger=log,
                        max_retries=10,
                        quite=quite)
                    if _last_successed:
                        matched, recall, precision = count_contained_intervals(test_sentences[next]["relations"], output["relations"])
                        total[0] += len(test_sentences[next]["relations"])
                        total[1] += len(output["relations"])
                        total[2] += matched
                        results[str(next)] = output
                        log(f"Output: {output}")
                        log(f"Time elapsed: {time.time() - current_time:.2f}s, Length: {len(output["relations"])}")
                        log(f"[{next}] Recall: {recall * 100:.2f}%, Precision: {precision * 100:.2f}%, Matched: {matched}/{len(test_sentences[next]["relations"])}")
                    else:
                        total[3] += 1
                        log(f"Time elapsed: {time.time() - current_time:.2f}s, Failed")
                    if _interrupted:
                        break
                if _interrupted:
                    log("\nExiting...")
                    break
                next += 1
                with open(checkpointfile, "w", encoding="utf-8") as f:
                    f.write(str(next))
        except KeyboardInterrupt:
            _interrupted = True
            log("\nExiting...")
        finally:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f)
            with open(checkpointfile, "w", encoding="utf-8") as f:
                log(f"Checkpoint saved at {next}")
                f.write(str(next))
            with open(workdir / "logs" / f"{get_time_str()}_{identifier}.log", "w", encoding="utf-8") as f:
                f.write("\n".join(logs))
    return _interrupted

tasks = []

for m in ["llama3.1:70b", "llama3.2:3b"]:
    for n in [3, 1]:
        for f in [False, True]:
            tasks.append({
                "model": m,
                "numofexps": n,
                "quite": False,
                "fixed": f,
            })

def main():
    for d in ["results", "record", "logs", "prompt" if __debug__ else None]:
        if d and not os.path.exists(workdir / d): os.mkdir(workdir / d)
        
    batch_size = 1 # int(input("Enter batch size: "))
    for index, task in enumerate(tasks):
        model = task["model"]
        nexps = task["numofexps"]
        quite = task["quite"]
        fixed = task["fixed"]
        embedder = ServerEmbedder("http://172.16.129.30:11434", model_name=model)
        # embedder = LocalEmbbeder(
            # model_name="google-bert/bert-base-uncased"
            # model_name='sentence-transformers/all-MiniLM-L6-v2'
        # )
        
        identifier = f"{embedder}_{model.replace(':', '-')}_{nexps}exps{'_fixed' if fixed else ''}_rel"
        title = f"Task {index+1}:: Embedder={embedder} Model={model} Examples={nexps}{',Fixed' if fixed else ''}"
        results_file = (workdir / "results" / f"{identifier}.json")
        results = json.loads(load_checkpoint(results_file, "{}"))
        _interrupted = evaluate(results_file, results, model, embedder, title, identifier=identifier, numofexps=nexps, fixed=fixed, quite=quite, batch_size=batch_size)
        if _interrupted:
            break

if __name__ == "__main__":
    main()