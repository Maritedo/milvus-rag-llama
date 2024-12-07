from lib.utility import *
from pymilvus import connections
from dotenv import load_dotenv
import os
# from openai import OpenAI

# # 加载环境变量
# load_dotenv()
# _api_key = os.getenv("DASHSCOPE_API_KEY")

# # 使用DashScope API的兼容模式
# client = OpenAI(
#     api_key=_api_key,  # 使用环境变量中的API Key
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # DashScope SDK的base_url
# )

def extract_spo(text, exps):
    # 根据示例数目生成示例部分
    examples = '；'.join([f"输入{i+1}：{exps[i]["input"]}，输出{i+1}：{exps[i]["output"]}" for i in range(numofexp)])
    # 不包含示例和SPO关系的提示词
    prompt = f'你是一个命名实体识别API，我将提供分词后的语句，你需要提取其中的命名实体及其索引范围，并进一步提取出句子中存在的关系并表示为SPO元组。作为API，你的输出应为严格的JSON格式而非Markdown，并且不能包含多余的自然语言，JSON包含两个字段，其中ner字段包含识别到的命名实体，relations字段包含识别到的关系。'
    # 加入 numofexp 个示例
    if len(exps) > 0:
        prompt += "示例输入如下："
        for exp in exps:
            prompt += f"""输入：{exp['sentence']}，输出：\{'ner':{exp['ner']},"relations":{1}}"""
    # 加入 SPO 关系限制
    
    completion = client.chat.completions.create(
        model="qwen-max-2024-09-19",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text}]
        )
    json_result = json.loads(completion.model_dump_json())
    input_token = json_result['usage']['prompt_tokens']
    output_token = json_result['usage']['completion_tokens']

    output_text = json_result['choices'][0]['message']['content']
    return json.loads(output_text)  # 假设模型按指定格式返回 JSON 数据

def compare_spo_lists(original_spo_list, generated_spo_list):
    """
    比对原始SPO列表和生成的SPO列表，并计算匹配百分比。
    返回：匹配百分比、缺失项数量、新增项数量
    """
    original_set = set(spo['predicate'] for spo in original_spo_list)
    generated_set = set(spo['p'] for spo in generated_spo_list)
    
    matched = original_set & generated_set
    missing = original_set - generated_set
    additional = generated_set - original_set
    
    match_percentage = (len(matched) / len(original_set)) * 100 if original_set else 100
    return match_percentage, len(missing), len(additional)

if __name__ == "__main__":
    try:
        while True:
            sentence=input("Enter a sentence: ")
            if sentence == "exit":
                break
            search_result = search(sentence)
            print(f"\nTop-3 search results for: '{sentence}'")
            for i, result in enumerate(search_result[0]):  # search_result[0] 对应第一个查询向量的结果
                print(f"Result {i+1}:")
                # print(f"  ID: {result.id}")
                print(f"  Distance: {result.distance}")
                # print(f"  Sentence: {result.entity.get('sentence')}")
                # print(f"  NER: {json.loads(result.entity.get('ner'))}")
                # print(f"  Relations: {json.loads(result.entity.get('relations'))}")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Disconnecting from Milvus...")
        collection.release()
        connections.disconnect("default")