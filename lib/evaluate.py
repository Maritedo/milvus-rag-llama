from . import filter_input

def compare_tag(a, b):
    return filter_input(a).lower().replace(' ', '').replace('-', '') == filter_input(b).lower().replace(' ', '').replace('-', '')

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
        item_g = _generated[min(index_g, len_g - 1)][1:]
        if item_t[0] < item_g[0]: # 生成项起始索引不被目标项包含
            index_t += 1
            continue
        if item_t[1] > item_g[1]: # 生成项终止索引不被目标项包含
            index_g += 1
            continue
        if compare_tag(item_t[2], item_g[2]):
            matched += 1
        if index_t < len_t - 1:
            index_t += 1
        elif index_g < len_g - 1:
            index_g += 1
        else:
            break
    return (matched, matched / len_t if len_t > 0 else -1, matched / len_g if len_g > 0 else -1)

def count_contained_intervals(target, generated):
    count = 0
    for _target in target:
        if len(_target) == 2:
            a_text, a_tag = _target
        elif len(_target) == 3:
            a_start, a_end, a_tag = _target
        elif len(_target) == 4:
            a_text, a_start, a_end, a_tag = _target
        else:
            raise ValueError("Invalid target item")
        for b_text, b_tag in generated:
            if (a_text in b_text) and compare_tag(a_tag, b_tag):
                count += 1
                break
    return (count, count / len(target) if len(target) > 0 else -1, count / len(generated) if len(generated) > 0 else -1)

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

if __name__ == "__main__":
    # ls = get_all(0, 5)
    # import os
    # errors = 0
    # numbers = 0
    # os.system("clear")
    # print(f"{errors}/{numbers}")
    # try:
    #     for i in ls:
    #         for j in ls:
    #             numbers += 1
    #             aa = match_list(i, j)[0]
    #             bb = count_contained_intervals(i, j)
    #             if aa != bb:
    #                 errors += 1
    #             os.system("clear")
    #             print(f"{errors}/{numbers} {aa}:{bb}")
    # except KeyboardInterrupt:
    #     exit(0)
    pass