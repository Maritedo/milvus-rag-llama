## 探究笔记
数据集的索引从0开始，并且切分时没有去除标点符号，即：
```python
str.split()
```

每一个句子的结果中，实体形式为三元组，关系形式为五元组，实体形式为 `[startIndex, endIndex, classification]`，关系形式为 `[startIndexOfSubj, endIndexOfSubj, startIndexOfObj, endIndexOfObj, classification]`