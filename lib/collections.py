from . import collection, collection_name, embedding_model
import json

def get_sentence_hash(sentence):
    return hash(sentence)  # 生成一个整数哈希值

def insert_sentence(sentences, ner, relations):
    # 转换句子为嵌入
    embeddings = embedding_model.encode(sentences).tolist()

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
        relations_json,    # relations 字段（JSON 字符串）
    ]
    insert_result = collection.insert(data)
    collection.flush()
    print(f"Inserted {len(insert_result.primary_keys)} rows into '{collection_name}'.")

def query(sentence, n = 3):
    collection.load()  # 加载数据到内存
    query_embedding = embedding_model.encode([sentence]).tolist()
    search_params = {
        "metric_type": "L2",
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