from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 连接到 Milvus
# local: 127.0.0.1
# whut : 172.16.129.30
connections.connect("default", host="172.16.129.30", port="19530")
print("Connected to Milvus.")

# 检查 Collection 是否已存在
collection_name = "ner_rag_sentences_train"
if collection_name in utility.list_collections():
    print(f"Collection '{collection_name}' already exists. Loading...")
    collection = Collection(name=collection_name)  # 加载现有 Collection
    # collection.drop()  # 删除 Collection
else:
    print(f"Collection '{collection_name}' does not exist. Creating...")
    # 定义 Collection 的 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="hash", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
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
        "metric_type": "L2",        # 相似度度量方式，例如 L2 距离
        "params": {"nlist": 128}    # 额外的参数设置
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Index created for '{collection_name}'.")