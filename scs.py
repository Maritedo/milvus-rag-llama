from pathlib import Path
import os
import json
from math import log10
from lib import filter_input

workdir = Path(os.getcwd())

fls = []
for root, dirs, files in os.walk(workdir / "results"):
    for file in files:
        if file.endswith(".json"):
            fls.append(file)
list_display_size = int(log10(len(fls))) + 1
# for index, file in enumerate(fls):
#     print(f'{index + 1:>{list_display_size}}. {file}')
selected_file = 0
while selected_file == None:
    print("Available files:")
    try:
        _selected_file = int(input("Select file: ")) - 1
        if _selected_file < 0 or _selected_file >= len(fls):
            raise Exception
        selected_file = _selected_file
        break
    except ValueError:
        print("Please enter a number.")
        continue
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception:
        print("Invalid number.")
        continue

print(f"Selected file: {fls[selected_file]}")

import os
if not os.path.exists(workdir / "statis"):
    os.mkdir(workdir / "statis")
outputfile = open(workdir / "statis" / f"scs_{fls[selected_file].replace(".json", ".txt")}", "w", encoding="utf-8")

with open(workdir / "results" / fls[selected_file], "r", encoding="utf-8") as f:
    generated_datas = json.load(f)

with open(workdir / "data" / "test_sentence.parsed.json", "r", encoding="utf-8") as f:
    target_datas = json.load(f)
    
def scs_calc(text1: str, text2: str, character_mode=False) -> int:
    text1 = text1.lower()
    text2 = text2.lower()
    if not character_mode:
        text1 = text1.split()
        text2 = text2.split()
    len1 = len(text1)
    len2 = len(text2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_length = 0

    # 填充 DP 表
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1[i - 1] == text2[j - 1]:  # 如果字符相等
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0

    return max_length**2/len1/len2, max_length

import numpy as np
from scipy.optimize import linear_sum_assignment

def scs_comp_list(expected: list[str], generated: list[str], character_mode=False) -> tuple:
    mat = [[0] * len(generated) for _ in range(len(expected))]
    for i, e in enumerate(expected):
        for j, g in enumerate(generated):
            mat[i][j] = scs_calc(filter_input(e[0]), filter_input(g[0]), character_mode=character_mode)[0]
    matrix = np.array(mat)
    m, n = matrix.shape
    p = min(m, n)
    print(matrix)
    # 构造一个方阵（填充负无穷）
    cost_matrix = np.full((max(m, n), max(m, n)), -1e10)
    
    # 将原矩阵的元素复制到新的矩阵中
    cost_matrix[:m, :n] = matrix
    print(cost_matrix)
    for i in range(m):
        for j in range(n):
            if i < p and j < p:
                cost_matrix[i, j] = matrix[i, j]
    # 使用线性分配算法求解最大权匹配
    # 求解最小化问题，所以取负号
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    selected_elements = [(row_ind[i], col_ind[i], matrix[row_ind[i], col_ind[i]]) for i in range(p)  if row_ind[i] < m and col_ind[i] < n]
    max_sum = sum(matrix[row_ind[i], col_ind[i]] for i in range(p)  if row_ind[i] < m and col_ind[i] < n)
    print(f"Selected elements: {selected_elements}")
    return max_sum, selected_elements

max_s =  10
for key in generated_datas.keys():
    if max_s == 0:
        break
    max_s -= 1
    print(f"Processing {key}...")
    outputfile.write(f"Processing {key}...\n")
    expected = target_datas[int(key)]["entities"]
    generated = generated_datas[str(key)]["entities"]
    score, _ = scs_comp_list(expected, generated)
    print(f"Gen: {[filter_input(i[0]) for i in expected]}")
    print(f"Tgt: {[i[0] for i in generated]}")
    print(f"Score: {score}")
    outputfile.write(f"Score: {score}\n")
    outputfile.write("\n")
    
    
outputfile.close()