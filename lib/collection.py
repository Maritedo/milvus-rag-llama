import json
import os

from dotenv import load_dotenv

from . import get_sentence_hash
from .embedder import Embedder

load_dotenv()

DEFAULT_HOST = os.getenv("MILVUS_HOST", "")
DEFAULT_PORT = os.getenv("MILVUS_PORT", "19530")

class NEREmbeddingColleciton:
    metric_type = "COSINE"
    index_type = "IVF_FLAT"
    def __init__(self) -> None:
        self.collection_name = None
        self.collection = None
        self.embedder = None
        
        
    def __create_database(self) -> None:
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
        # 定义 Collection 的 Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="hash", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedder.dimension()),
            FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="ner", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="relations", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, description="Sentence storage with embeddings, NER, and relations")
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection '{self.collection_name}' created.")
        # 创建索引
        index_params = {
            "index_type": self.index_type,  # 索引类型，例如 IVF_FLAT 或 HNSW
            "metric_type": self.metric_type,   # 相似度度量方式，例如 L2 距离
            "params": {"nlist": 128}   # 额外的参数设置
        }
        collection.create_index(field_name="embedding", index_params=index_params)


    def connect(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        from pymilvus import connections
        connections.connect("default", host=host, port=port)
        
        
    def init(self, embedder: Embedder):
        if self.embedder != embedder:
            self.embedder = embedder
            self.collection_name = f"ner_sentences_train_{embedder.name().split('/')[-1].replace('-', '_')}"
        else:
            return
        from pymilvus import Collection, utility
        if self.collection_name in utility.list_collections():
            print(f"Collection '{self.collection_name}' already exists. Loading...")
            self.collection = Collection(name=self.collection_name)
        else:
            print(f"Collection '{self.collection_name}' does not exist. Creating...")
            self.__create_database()
            print(f"Index created for '{self.collection_name}'.")


    def load(self) -> None:
        self.collection.load()
        
        
    def insert(self, data):
        insert_result = self.collection.insert(data)
        self.collection.flush()
        return insert_result


    def insert_sentence(self, sentences, ner, relations):
        data = [
            [get_sentence_hash(i) for i in sentences],   # hash 字段
            self.embedder.embbed(sentences),             # embedding 向量
            sentences,                                   # sentence 字段
            [json.dumps(i, indent=0) for i in ner],      # ner 字段（JSON 字符串）
            [json.dumps(i, indent=0) for i in relations] # relations 字段（JSON 字符串）
        ]
        return self.insert(data)
    
    
    def query(self, expr, output_fields):
        return self.collection.query(expr=expr, output_fields=output_fields)
    
        
    def search(self, sentences, limit=10, output_fields=["sentence", "ner", "relations"]):
        if type(sentences) == str:
            sentences = [sentences]
        self.collection.load()  # 加载数据到内存
        query_embedding = self.embedder.embbed(sentences)
        search_params = {
            "metric_type": self.metric_type,
            "params": { "nprobe": 10 }
        }
        search_result = self.collection.search(
            data=query_embedding,        # 查询向量
            anns_field="embedding",      # 检索的向量字段
            param=search_params,         # 搜索参数
            limit=limit,                 # 返回的前 n 个结果
            output_fields=output_fields  # 返回的附加字段
        )
        return search_result
    