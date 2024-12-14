def scs_evaluate(text1: str, text2: str, character_mode=False) -> int:
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

print(scs_evaluate("I am a student", "I am a teacher"))