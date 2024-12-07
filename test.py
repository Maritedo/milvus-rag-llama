def match_list(target: list[tuple[int, int, str]], generated: list[tuple[int, int, str]]) -> tuple[int, float, float]:
    """
    返回值: (matched, recall, precision)
    """
    matched = 0
    _target = sorted(target, key=lambda x: x[0])
    _generated = sorted(generated, key=lambda x: x[0])
    index_t = 0
    index_g = 0
    len_t = len(target)
    len_g = len(generated)
    while index_g < len_g and index_t < len_t:
        item_t = _target[min(index_t, len_t - 1)]
        item_g = _generated[min(index_g, len_g - 1)]
        if item_t[0] < item_g[0]: # 生成项起始索引不被目标项包含
            index_t += 1
            continue
        if item_t[1] > item_g[1]: # 生成项终止索引不被目标项包含
            index_g += 1
            continue
        # if item_t[2].lower().replace(' ', '') == item_g[2].lower().replace(' ', ''):
        matched += 1
        if index_t < len_t - 1:
            index_t += 1
        elif index_g < len_g - 1:
            index_g += 1
        else:
            break
    return (matched, matched / len_t, matched / len_g)
            

# match_list(
#     target=[(0, 1, "a"), (3, 5, "b")],
#     generated=[(1, 2, "a"), (3, 6, "c")]
# )

import itertools

def generate_partitions(a, b):
    """
    生成[a, b]区间所有可能的切分

    :param a: 区间起始点
    :param b: 区间结束点
    :return: 列表，包含从不切分到切分为b-a+1段的所有可能性
    """
    partitions = []
    for i in range(b - a + 2):  # 从0到b-a+1段
        if i == 0:
            partitions.append([a, b])
        else:
            steps = list(range(1, b - a + 1))
            for combination in itertools.combinations_with_replacement(steps, i):
                if sum(combination) == b - a:
                    partition = [a]
                    for step in combination:
                        partition.append(partition[-1] + step)
                    partitions.append(partition)
    return partitions

def count_contained_intervals(A, B):
    """
    计算A中可被B中列表区间包含的区间数量

    :param A: 列表，包含互不重合的若干区间和标识符（元组）
    :param B: 列表，包含互不重合的若干区间和标识符（元组）
    :return: 可以被B中任意区间所包含的A中的区间数量
    """
    # 初始化计数器
    count = 0

    # 遍历A中的每个区间和标识符
    for a_start, a_end in A:
        # 遍历B中的每个区间和标识符
        for b_start, b_end in B:
            # 判断是否包含且标识符相等
            if b_start <= a_start and a_end <= b_end:#and a_id == b_id:
                # 如果包含，则计数加1
                count += 1
                # break跳出内层循环，因为已经找到一个包含关系
                break

    return count

def get_all(a, b):
    res = []
    for i in range(0, 2**(b-a)):
        slices = []
        start = a
        for j in range(a, b + 1):
            if not i & (1 << (j - a)):
                slices.append([start, j])
                start = j + 1
        res.append(slices)
        # print("".join(reversed(f"{i:0{b-a}b}")), slices)
    return res
            
ls = get_all(0, 5)
import os
errors = 0
numbers = 0
os.system("clear")
print(f"{errors}/{numbers}")
try:
    for i in ls:
        for j in ls:
            numbers += 1
            aa = match_list(i, j)[0]
            bb = count_contained_intervals(i, j)
            if aa != bb:
                errors += 1
            os.system("clear")
            print(f"{errors}/{numbers} {aa}:{bb}")
except KeyboardInterrupt:
    exit(0)