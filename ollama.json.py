from ollama import ChatResponse, Client
from pydantic import BaseModel

class NER_entities(BaseModel):
    entity: str
    entity_type: str

class NER_relations(BaseModel):
    A: str
    relation_type: str
    B: str

class NER_response(BaseModel):
    entities: list[NER_entities]
    relations: list[NER_relations]

ollam_client = Client(host="http://172.16.129.30:11434")

schema_ner = """所有可能的实体类型:
1.Task: 要实现的应用、要解决的问题、要构建的系统
2.Method: 使用的方法、模型、系统、工具，系统或框架的组件
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

system = f"""作为自然语言处理服务API，你会接收到包含句子输入，你需要从输入的句子中提取实体，并找出提取出的实体之间的关系。你给出的结果必须是纯JSON字符串。不要给出代码或解释说明。
{schema_ner}
{schema_rel}
一些输入和应给出的输出示例如下：
输入1："The agreement in question involves number in nouns and reflexive pronouns and is syntactic rather than semantic in nature because grammatical number in English , like grammatical gender in languages such as French , is partly arbitrary ."
输出1：{{"entities": [["agreement", "Generic"], ["nouns", "OtherScientificTerm"], ["reflexive pronouns","OtherScientificTerm"], ["grammatical number", "OtherScientificTerm"], ["English", "Material"], ["grammatical gender", "OtherScientificTerm"], ["languages", "Material"], ["French", "Material"]], "relations": [["nouns", "reflexive pronouns", "CONJUNCTION"], ["grammatical gender", "languages", "FEATURE-OF"], ["French", "languages", "HYPONYM-OF"]]}}
在提取实体以及找出提取出的实体的关系时，应提取尽可能多的结果，并且不应对内容进行改写。等待输入并给出结果...
"""

user = "It has also been studied in the framework of Japanese information extraction -LRB- -LSB- 3 -RSB- -RRB- in recent years ."

def get1():
    response: ChatResponse = ollam_client.chat(
        model="llama3.1:70b",
        stream=False,
        format=NER_response.model_json_schema(),
        messages=
        [{
            'role': 'system',
            'content': system
        },
        {
            'role': 'user',
            'content': user
        }]
    )

import requests
def get2():
    api_url = "http://172.16.129.30:11434/api/chat"
    payload = {
        "model": "llama3.1:70b",
        "system": system,
        "prompt": user,
        "stream": False,
        "format": NER_response.model_json_schema()
    }
    response = requests.post(api_url, json=payload)
    return response.json()

response = get2()
print(response)