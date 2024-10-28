def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False


def algin(title_key, max_english):
    chinese_count = 0
    english_count = 0
    for j in str(title_key):
        if is_Chinese(j):
            chinese_count = chinese_count + 1
        else:
            english_count = english_count + 1

    temp = max_english - english_count
    while temp > 0:
        title_key = title_key + ' '
        temp = temp - 1
    title_key = title_key.ljust(7, chr(12288))
    # print(title_key + '-')
    return title_key


if __name__ == '__main__':
    algin("a一二三", 3)
    algin("aa一二三", 3)
    algin("aaa一二三", 3)
    algin("a一二三aa", 3)