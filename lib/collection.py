import json
from . import get_sentence_hash
from .embedder import Embedder

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
