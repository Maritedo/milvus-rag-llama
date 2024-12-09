import sqlite3
from typing import List, Tuple

class QueryCacheOld:
    def __init__(self, algorithm_name: str, db_path: str = "cache.db"):
        """
        初始化缓存对象，基于算法名动态创建或使用数据表
        :param algorithm_name: 算法名，用于区分不同的结果缓存表
        :param db_path: SQLite 数据库文件路径
        """
        self.algorithm_name = algorithm_name
        self.table_name = f"query_cache_{algorithm_name}.old"
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_table()

    def _initialize_table(self):
        """
        初始化表结构，如果表不存在则创建
        """
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS [{self.table_name}] (
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
            f"INSERT OR REPLACE INTO [{self.table_name}] (input_index, rank, data_index, similarity) VALUES (?, ?, ?, ?)",
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
            f"SELECT data_index, similarity FROM [{self.table_name}] WHERE input_index = ? ORDER BY rank",
            (input_index,)
        )
        return self.cursor.fetchall()

    def __del__(self):
        """
        关闭数据库连接
        """
        self.conn.close()

class QueryCache:
    def __init__(self, algorithm_name: str, db_path: str = "cache.db"):
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

    def insert_results(self, input_index: int, results: List[Tuple[int, int, float]]):
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

    def fetch_results(self, input_index: int) -> List[Tuple[int, int, float]]:
        """
        从缓存中查询结果
        :param input_index: 查询索引
        :return: 查询结果，列表，包含 (数据索引, 向量索引, 相似度) 的元组
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