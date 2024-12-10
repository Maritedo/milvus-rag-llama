import sqlite3
import time
from typing import List, Tuple
from lib.utils import search, embedder
import json
from pathlib import Path
import os

workdir = Path(os.getcwd())

class QueryCache:
    def __init__(self, algorithm_name: str, db_path: str = workdir / 'cache' / 'cache.db'):
        """
        初始化缓存对象，基于算法名动态创建或使用数据表
        :param algorithm_name: 算法名，用于区分不同的结果缓存表
        :param db_path: SQLite 数据库文件路径
        """
        self.algorithm_name = algorithm_name
        self.table_name = f"query_cache_{algorithm_name}"
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_table()

    def _initialize_table(self):
        """
        初始化表结构，如果表不存在则创建
        """
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            input_index INTEGER,
            rank INTEGER,
            data_index INTEGER,
            similarity REAL,
            PRIMARY KEY (input_index, rank)
        )
        """)
        self.conn.commit()

    def insert_results(self, input_index: int, results: List[Tuple[int, float]]):
        """
        插入查询结果，如果已存在则替换
        :param input_index: 查询索引
        :param results: 查询结果，列表，包含 (数据索引, 相似度) 的元组
        """
        self.cursor.executemany(
            f"INSERT OR REPLACE INTO {self.table_name} (input_index, rank, data_index, similarity) VALUES (?, ?, ?, ?)",
            [(input_index, rank, data_index, similarity) for rank, (data_index, similarity) in enumerate(results)]
        )
        self.conn.commit()

    def fetch_results(self, input_index: int) -> List[Tuple[int, float]]:
        """
        从缓存中查询结果
        :param input_index: 查询索引
        :return: 查询结果，列表，包含 (数据索引, 相似度) 的元组
        """
        self.cursor.execute(
            f"SELECT data_index, similarity FROM {self.table_name} WHERE input_index = ? ORDER BY rank",
            (input_index,)
        )
        return self.cursor.fetchall()

    def __del__(self):
        """
        关闭数据库连接
        """
        self.conn.close()

if __name__ == "__main__":
    cache = QueryCache(embedder.name())
    with open(Path(__file__).parent / 'data' / 'train_sentence.json', "r") as f:
        json_data = json.load(f)
        train_sentences = [item["sentence"] for item in json_data]
        
    with open(Path(__file__).parent / 'data' / 'test_sentence.json', "r") as f:
        json_data = json.load(f)
        test_sentences = [item["sentence"] for item in json_data]
    next = 0
    with open(workdir / 'record' / ('cache_' + embedder.name() + '.next'), 'a+') as f:
        fc = f.read()
        if fc and fc.strip().isdigit():
            next = int(fc)
    print("Modes: 1. Fix, 2. Batch")
    while True:
        mode = input("Input a mode: ")
        if mode.isdigit() and int(mode) in [1, 2]:
            break
    if mode == "1":
        while True:
            try:
                index = int(input("Enter index: "))
                if 0 <= index < len(test_sentences):
                    current_time = time.time()
                    print(f"Processing {index}...")
                    result = search(test_sentences[index], n=10)[0]
                    tobeinserted = [(train_sentences.index(row.entity.get("sentence")), row.distance) for row in result]
                    print(*tobeinserted, sep="\n")
                    if input("Insert? (y/n): ").lower() != "y":
                        continue
                    cache.insert_results(index, tobeinserted)
                    print(f"Time elapsed: {time.time() - current_time:.2f}s")
            except KeyboardInterrupt:
                print("\nExiting...")
                exit(0)
            except Exception as e:
                print(e)
                exit(1)
                
    batch_size = int(input("Enter batch size: "))
    while True:
        if next >= len(test_sentences):
            break
        current_time = time.time()
        print(f"Processing {next} to {min(next + batch_size, len(test_sentences))}...")
        items = test_sentences[next:next+batch_size]
        results = search(items, n=10)
        for offset, result in enumerate(results):
            # print(f"Input {next+offset}: {items[offset]}")
            # print("Results:")
            for i, row in enumerate(result):
                print(f"  {train_sentences.index(row.entity.get("sentence")):<4}. ({row.distance}) {row.entity.get('sentence')}")
            tobeinserted = [(train_sentences.index(row.entity.get("sentence")), row.distance) for row in result]
            cache.insert_results(next + offset, tobeinserted)
            # input("Press Enter to continue...")
        next = min(next + batch_size, len(test_sentences))
        with open(workdir / 'record' / ('cache_' + embedder.name() + '.next'), 'w') as f:
            f.write(str(next))
        print(f"Time elapsed: {time.time() - current_time:.2f}s")