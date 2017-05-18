# encoding: utf-8

# 获取停用词表
def get_stop_words():
    words_file = open('stop_words.txt', 'rb')
    words_list = []
    for line in words_file:
        words_list.append(line.decode('utf-8').strip())
    return words_list

if __name__ == '__main__':
    list = get_stop_words()
    for i in list:
        print i[0],
