"""
工具函数模块 (utils.py)

这个模块提供了一些文本处理相关的工具函数，主要用于：
1. 判断字符是否为中文字符
2. 对齐中英文混合文本（用于格式化输出）

使用场景：
- 文本格式化
- 中英文混合文本的对齐处理
"""

def is_Chinese(ch):
    """
    判断单个字符是否为中文字符
    
    这个函数通过Unicode编码范围来判断字符是否为中文。
    中文字符的Unicode编码范围是 \u4e00 到 \u9fff。
    
    参数:
        ch: 要判断的单个字符
        
    返回:
        True: 是中文字符
        False: 不是中文字符
        
    示例:
        is_Chinese('中')  # 返回 True
        is_Chinese('a')   # 返回 False
        is_Chinese('1')   # 返回 False
    """
    # Unicode编码范围：\u4e00 到 \u9fff 是中文字符的编码范围
    # 这个范围包含了大部分常用汉字
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False


def algin(title_key, max_english):
    """
    对齐中英文混合文本
    
    这个函数用于格式化中英文混合的文本，使其在显示时对齐。
    由于中文字符和英文字符的宽度不同（中文通常占2个字符宽度），
    需要特殊处理才能实现对齐。
    
    参数:
        title_key: 要对齐的文本（可能包含中文和英文）
        max_english: 最大英文字符数（用于计算对齐所需的空格数）
        
    返回:
        对齐后的文本字符串
        
    工作原理：
    1. 统计文本中的中文字符数和英文字符数
    2. 计算需要添加的空格数（max_english - 英文字符数）
    3. 添加空格使英文字符数达到max_english
    4. 使用全角空格（chr(12288)）填充到指定宽度，实现对齐
        
    示例:
        algin("a一二三", 3)    # 返回对齐后的文本
        algin("aa一二三", 3)   # 返回对齐后的文本
    """
    # 初始化计数器
    chinese_count = 0  # 中文字符数
    english_count = 0  # 英文字符数
    
    # 遍历文本中的每个字符，统计中英文数量
    for j in str(title_key):
        if is_Chinese(j):
            chinese_count = chinese_count + 1
        else:
            english_count = english_count + 1

    # 计算需要添加的空格数
    # 如果英文字符数少于max_english，需要添加空格
    temp = max_english - english_count
    # 添加半角空格（用于英文字符对齐）
    while temp > 0:
        title_key = title_key + ' '
        temp = temp - 1
    
    # 使用全角空格填充到指定宽度（7个字符）
    # chr(12288) 是全角空格的Unicode编码
    # ljust(7, chr(12288)) 在右侧填充全角空格，使总宽度为7
    # 全角空格与中文字符宽度相同，可以实现对齐效果
    title_key = title_key.ljust(7, chr(12288))
    return title_key


# 测试代码（当直接运行此文件时执行）
if __name__ == '__main__':
    # 测试不同情况下的对齐效果
    algin("a一二三", 3)
    algin("aa一二三", 3)
    algin("aaa一二三", 3)
    algin("a一二三aa", 3)