import requests
import json
from math import log10
from lib import get_sentence_hash

# filedir = workdir / "data"
# load_dotenv(workdir / ".env")

class Embedder:
    def __init__(self) -> None:
        raise NotImplementedError
    def embbed(self, sentence: str):
        pass
    def name(self):
        pass
    def dimension(self):
        pass
    
class LocalEmbbeder(Embedder):
    def __init__(self, model_name) -> None:
        self.model_name = model_name.split("/")[-1].replace("-", "_")
        self.__loaded = False
        self.model = None
    
    def __lazy_load(self):
        if not self.__loaded:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.__loaded = True
    
    def embbed(self, sentences):
        self.__lazy_load()
        if type(sentences) == str:
            sentences = [sentences]
        return self.model.encode(sentences).tolist()
    
    def name(self):
        return self.model_name
    
    def dimension(self):
        self.__lazy_load()
        return self.model.get_sentence_embedding_dimension()

def get_size_readable(size):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

class ServerEmbedder(Embedder):
    def __interactive_choose(self, server_url):
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
                    break
        else: raise Exception(f"Failed to get models from {api_url}")
        return (model, model_name)
    
    def __init__(self, server_url: str, model_name: str | None = None) -> None:
        if model_name is None:
            (model, model_name) = self.__interactive_choose(server_url)
            print(f"Selected model: {model['name']} ({get_size_readable(model['size'])})")
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

def _choose_local():
    models_list = [
        "google-bert/bert-base-uncased",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    print("Available models:")
    for index, model in enumerate(models_list):
        print(f"{index+1}. {model}")
    while True:
        index_str = input("Choose a model: ")
        if not index_str.isdigit():
            continue
        index = int(index_str) - 1
        if 0 <= index < len(models_list):
            model = models_list[index]
            print(f"Selected model: {model}")
            return SentenceTransformer(model)

def _choose_server():
    server_url = input("Enter server URL:")
    if not server_url:
        server_url = "http://172.16.129.30:11434"
    return ServerEmbedder(server_url=server_url, model_name=None)

def choose_embedder():
    embedder = None
    while embedder is None:
        i = input("Choose an embedder (local/server): ").lower()
        if not i in ["local", "server"]:
            continue
        if i == "local":
            embedder = _choose_local()
        else:
            embedder = _choose_server()
    return embedder

class NEREmbeddingColleciton:
    def __init__(self, embedder: Embedder) -> None:
        self.model_name = embedder.name()
        self.collection_name = f"ner_sentences_train_{self.model_name.split('/')[-1].replace('-', '_')}"
        self.collection = None
        self.embedder = embedder
    
    
    def __create_database(self) -> None:
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
        # 定义 Collection 的 Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="hash", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedder.dimension),
            FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="ner", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="relations", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, description="Sentence storage with embeddings, NER, and relations")
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection '{self.collection_name}' created.")
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",  # 索引类型，例如 IVF_FLAT 或 HNSW
            "metric_type": "COSINE",   # 相似度度量方式，例如 L2 距离
            "params": {"nlist": 128}   # 额外的参数设置
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    
    
    def connect(self):
        from pymilvus import connections
        # 连接到 Milvus
        # locals: 127.0.0.1
        # school: 172.16.129.30
        connections.connect("default", host="172.16.129.30", port="19530")
        print("Connected to Milvus.")
        
    def init(self):
        from pymilvus import Collection, utility
        if self.collection_name in utility.list_collections():
            print(f"Collection '{self.collection_name}' already exists. Loading...")
            self.collection = Collection(name=self.collection_name)
        else:
            print(f"Collection '{self.collection_name}' does not exist. Creating...")
            self.__create_database()
            print(f"Index created for '{self.collection_name}'.")
    
    def load(self):
        self.collection.load()
    
    
    def insert(self, data):
        insert_result = self.collection.insert(data)
        self.collection.flush()
        return insert_result
    

    def insert_sentence(self, sentences, ner, relations):
        embeddings = self.embedder.embbed(sentences)
        
        ner_json = [json.dumps(i, indent=0) for i in ner]
        relations_json = [json.dumps(i, indent=0) for i in relations]

        data = [
            [get_sentence_hash(i) for i in sentences], # hash 字段
            embeddings,                                # embedding 向量
            sentences,                                 # sentence 字段
            ner_json,                                  # ner 字段（JSON 字符串）
            relations_json                             # relations 字段（JSON 字符串）
        ]
        insert_result = self.collection.insert(data)
        self.collection.flush()
        return insert_result

    
    def query(self, expr, output_fields):
        return self.collection.query(expr=expr, output_fields=output_fields)
    
    def search(self, sentences, anns_field, param, limit, output_fields=["sentence", "ner", "relations"]):
        if type(sentences) == str:
            sentences = [sentences]
        self.collection.load()  # 加载数据到内存
        query_embedding = self.embedder.embbed(sentences)
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 10
            }
        }
        search_result = self.collection.search(
            data=query_embedding,        # 查询向量
            anns_field="embedding",      # 检索的向量字段
            param=search_params,         # 搜索参数
            limit=limit,                 # 返回的前 n 个结果
            output_fields=output_fields  # 返回的附加字段
        )
        return search_result
