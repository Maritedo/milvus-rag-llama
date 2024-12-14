import os
import json
import requests
from flask import Flask, send_from_directory, request, jsonify, send_file
from dotenv import load_dotenv
from pathlib import Path

from lib.embedder import ServerEmbedder
from lib.collection import NEREmbeddingColleciton
from lib import filter_input, parse_input

workdir = Path(__file__).parent

_ = load_dotenv()
SERVER_PORT = os.getenv('SERVER_PORT', 18080)
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', 19530)

app = Flask(__name__, static_url_path='/static', static_folder='public')

collection_srv = NEREmbeddingColleciton()
collection_srv.connect()
embedder = ServerEmbedder(server_url=OLLAMA_URL, model_name="llama3.1:70b")
collection_srv.init(embedder)

with open(workdir / "data" / "train_sentence.parsed.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    train_data_sci: list = json_data
    train_sentences_sci: list = [item["sentence"] for item in train_data_sci]

def get_completion_sci(sentence, examples, model):
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
    def remove_mark(element):
        filtered = {
            "entities": [[filter_input(ent[0]), ent[1]] for ent in element['entities']],
            "relations": [[filter_input(rel[0]), filter_input(rel[1]), rel[2]] for rel in element['relations']]
        }
        return json.dumps(filtered)
    system = f"""你是一个命名实体抽取专家，你需要从输入的句子中提取尽可能多的实体，并找出提取出的实体之间的关系（若存在）。不要给出代码或解释说明。
{schema_ner}
{schema_rel}
一些输入和应给出的输出示例如下（format: {{"entities": [[entity1, etype], ...], "relations": [[entityA, entityB, rtype], ...]}}）：
{"\n".join([f'Input {i+1}: {filter_input(train_data_sci[item[0]]["sentence"])}\nOutput {i+1}: {remove_mark(train_data_sci[item[0]])}' for i, item in enumerate(examples)])}
在提取实体以及找出提取出的实体的关系时，应提取尽可能多的结果，并且不应对内容进行改写。等待输入并给出结果..."""
    prompt = sentence
    api_url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    response = requests.post(api_url, json=payload)
    return response.json()

def get_completion_general(sentence, model):
    schema_ner = """可能的实体类型（示例，名称以列表中为准，列表中未出现的自行选择合适的类型名）: 'Date', 'Number', 'Text', '人物', '企业', '企业/品牌', '作品', '历史人物', '国家', '图书作品', '地点', '奖项', '娱乐人物', '学校', '影视作品', '文学作品', '机构', '歌曲', '电视综艺', '音乐专辑'"""
    schema_rel = """可能的关系类型（以SPO三元组形式给出）:
(企业, 注册资本, Number)
(人物, 丈夫, 人物)
(人物, 毕业院校, 学校)
(机构, 简称, Text)
(文学作品, 主角, 人物)
(电视综艺, 主持人, 人物)
(影视作品, 制片人, 人物)
(娱乐人物, 饰演, 人物)
(人物, 父亲, 人物)
(影视作品, 导演, 人物)
(机构, 成立日期, Date)
(人物, 祖籍, 地点)
(影视作品, 改编自, 作品)
(影视作品, 主题曲, 歌曲)
(影视作品, 编剧, 人物)
(企业/品牌, 代言人, 人物)
(影视作品, 主演, 人物)
(企业, 总部地点, 地点)
(历史人物, 朝代, Text)
(娱乐人物, 获奖, 奖项)
(歌曲, 歌手, 人物)
(娱乐人物, 配音, 人物)
(电视综艺, 嘉宾, 人物)
(人物, 母亲, 人物)
(影视作品, 票房, Number)
(人物, 国籍, 国家)
(人物, 妻子, 人物)
(歌曲, 所属专辑, 音乐专辑)
(歌曲, 作曲, 人物)
(企业, 创始人, 人物)
(影视作品, 出品公司, 企业)
(图书作品, 作者, 人物)
(历史人物, 号, Text)
(企业, 董事长, 人物)
(歌曲, 作词, 人物)
(影视作品, 上映时间, Date)"""
    system = f"""你是一个命名实体及关系抽取专家，你需要从输入的句子中提取尽可能多的实体，并找出提取出的实体之间的关系（若存在）。不要给出代码或解释说明。
{schema_ner}
{schema_rel}
你的输出形式应为：{{"entities": [[entity1, etype], ...], "relations": [[entityA, entityB, rtype], ...]}}，即包含entities和relations两个字段的JSON对象，entities字段为一个列表，每个元素为一个列表，包含实体名和实体类型；relations字段为一个列表，每个元素为一个列表，包含两个实体名和关系类型。
"""
    prompt = sentence
    api_url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    response = requests.post(api_url, json=payload)
    return response.json()

# 静态文件服务
@app.route('/', methods=['GET'])
def serve_home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/ner/extract', methods=['POST'])
def extract():
    payload = json.loads(request.get_data().decode('utf-8'))
    sentence = payload.get('sentence', '')
    if not sentence:
        return jsonify({
            'error': "Parameter 'sentence' cannot be empty"
        }), 400
    field = payload.get('field', 'general')
    if field not in ['general', 'sci']:
        return jsonify({
            'error': "Parameter 'field' is invalid"
        }), 400
    model = payload.get('model', '')
    if not model:
        return jsonify({
            'error': "Parameter 'model' cannot be empty"
        }), 400
    try:
        if field == "sci":
            examples = collection_srv.search(sentence, limit=3, output_fields=["sentence"])[0]
            indexed = [(train_sentences_sci.index(row.entity.get("sentence")), row.distance) for row in examples]
            completion = get_completion_sci(sentence, indexed, model)
            print(completion["response"])
            return jsonify(completion["response"])
        elif field == "general":
            completion = get_completion_general(sentence, model)
            print(completion["response"])
            return jsonify(completion["response"])
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(SERVER_PORT))