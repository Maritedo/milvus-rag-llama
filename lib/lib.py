import requests
import json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from math import log10

class Embbeder:
    def __init__(self) -> None:
        pass
    def embbed(self, sentence):
        pass
    def name(self):
        pass
    def dimension(self):
        pass
    
class LocalEmbbeder(Embbeder):
    def __init__(self, model_name) -> None:
        self.model_name = model_name.split("/")[-1].replace("-", "_")
        self.model = SentenceTransformer(model_name)
    
    def embbed(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        return self.model.encode(sentences).tolist()
    
    def name(self):
        return self.model_name
    
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()

def get_size_readable(size):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

class ServerEmbbeder(Embbeder):
    def __init__(self, server_url: str, model_name: str | None = None) -> None:
        if model_name is None:
            api_url = f"{server_url}/api/tags"
            response = requests.get(api_url)
            if response.status_code == 200:
                models = response.json()["models"]
                print(f"Available models ({len(models)} in total):")
                if len(models) == 0:
                    print("No models available.")
                    raise Exception("No available models")
                display_len = int(log10(len(models))) + 1
                max_name_len = max([len(model['name']) for model in models])
                for index, model in enumerate(models):
                    print(f"{1+index:{display_len}d}. {model['name']:<{max_name_len}} ({get_size_readable(model['size'])})")
                print()
                while True:
                    index_str = input(f"Choose a model: ")
                    if not index_str.isdigit():
                        continue
                    index = int(index_str) - 1
                    if 0 <= index < len(models):
                        model = models[index]
                        model_name = model['name']
                        print(f"Selected model: {model['name']} ({get_size_readable(model['size'])})")
                        break
            else: raise Exception(f"Failed to get models from {api_url}")
        self.model_name = model_name
        self.server_url = server_url
        self.__dimension = None
    
    def embbed(self, sentences):
        if type(sentences) == str:
            sentences = [sentences]
        api_url = f"{self.server_url}/api/embed"
        payload = {
            "input": sentences,
            "model": self.model_name,
            "seed": 42
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
    
    def dimension(self):
        if self.__dimension is not None:
            return self.__dimension
        api_url = f"{self.server_url}/api/show"
        payload = {
            "model": self.model_name
        }
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            details = response.json()
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        self.__dimension = details["model_info"]["llama.embedding_length"]
        return self.__dimension
    
    def name(self):
        return self.model_name.split(":")[0].replace(".", "_")

class NEREmbeddingColleciton:
    def __init__(self, collection_name) -> None:
        self.collection_name = collection_name
        self.collection = None
    
    def __init__(self) -> None:
        self.collection_name = "ner_sentences_train"
        self.collection = None
    
    def connect(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="hash", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="ner", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="relations", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, description="Sentence storage with embeddings, NER, and relations")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection '{self.collection_name}' created.")
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",      # 索引类型，例如 IVF_FLAT 或 HNSW
            "metric_type": "COSINE",      # 相似度度量方式，余弦相似度
            "params": {"nlist": 128}               # 额外的参数设置
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Index created for '{self.collection_name}'.")
    
    def load(self):
        self.collection.load()
    
    def insert(self, data):
        insert_result = collection.insert(data)
        collection.flush()
        print(f"Inserted {len(insert_result.primary_keys)} rows into '{collection_name}'.")
    
    def query(self, expr, output_fields):
        self.collection.query()
        return self.collection.query(expr=expr, output_fields=output_fields)



# model_name = 'google-bert/bert-base-uncased'
# model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# embedding_model = SentenceTransformer(model_name)
# dimension = embedding_model.get_sentence_embedding_dimension()

embbeder = ServerEmbbeder("http://172.16.129.30:11434", model_name=None)
# embbeder = LocalEmbbeder("google-bert/bert-base-uncased")
model_name = embbeder.name()
dimension = embbeder.dimension()
print(f"Model: {model_name}, Dimension: {dimension}")

# 连接到 Milvus
# locals: 127.0.0.1
# school: 172.16.129.30
connections.connect("default", host="172.16.129.30", port="19530")
print("Connected to Milvus.")

# 检查 Collection 是否已存在
collection_name = f"ner_sentences_train_{model_name.split('/')[-1].replace('-', '_')}"
if collection_name in utility.list_collections():
    print(f"Collection '{collection_name}' already exists. Loading...")
    collection = Collection(name=collection_name)  # 加载现有 Collection
    # print(f"Collection length: {collection.num_entities}")
    # if input("Do you want to drop this collection? (y/n): ").lower() == "y":
    #     collection.drop()  # 删除 Collection
    # exit()
else:
    print(f"Collection '{collection_name}' does not exist. Creating...")
    # 定义 Collection 的 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="hash", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="ner", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="relations", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Sentence storage with embeddings, NER, and relations")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",  # 索引类型，例如 IVF_FLAT 或 HNSW
        "metric_type": "COSINE",        # 相似度度量方式，例如 L2 距离
        "params": {"nlist": 128}    # 额外的参数设置
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Index created for '{collection_name}'.")

def get_sentence_hash(sentence):
    return hash(sentence)  # 生成一个整数哈希值

def insert_sentence(sentences, ner, relations):
    # 转换句子为嵌入
    # embeddings = embedding_model.encode(sentences).tolist()
    embeddings = embbeder.embbed(sentences)
    
    # 转换 NER 和关系数据为 JSON 字符串
    ner_json = [json.dumps(i, indent=0) for i in ner]
    relations_json = [json.dumps(i, indent=0) for i in relations]

    # 插入数据到 Collection
    data = [
        # None,              # sentence_id: None 表示自动生成主键
        [get_sentence_hash(i) for i in sentences],                # hash 字段
        embeddings,        # embedding 向量
        sentences,         # sentence 字段
        ner_json,          # ner 字段（JSON 字符串）
        relations_json    # relations 字段（JSON 字符串）
    ]
    insert_result = collection.insert(data)
    collection.flush()
    print(f"Inserted {len(insert_result.primary_keys)} rows into '{collection_name}'.")

def query(sentences, n = 3):
    if type(sentences) == str:
        sentences = [sentences]
    collection.load()  # 加载数据到内存
    query_embedding = embbeder.embbed(sentences)
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "nprobe": 10
        }
    }
    search_result = collection.search(
        data=query_embedding,        # 查询向量
        anns_field="embedding",      # 检索的向量字段
        param=search_params,         # 搜索参数
        limit=n,                     # 返回的前 n 个结果
        output_fields=["sentence", "ner", "relations"],  # 返回的附加字段
    )
    return search_result
