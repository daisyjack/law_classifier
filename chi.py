# coding: utf-8
import cPickle as pickle

import utils

# 计算词的CHI，得到CHI由大到小排序的列表
def get_chi_wordlist(datas, num, key):
    # file_name_all = '/home/ren/law_crawler/data_law_tf/data_law_tf_all.pkl'
    # in_file = open(file_name_all, 'rb')
    # datas = pickle.load(in_file)
    # in_file.close()
    # a = ['我的','的','我的' ]
    # print a.count('我')
    # print '---'
    # datas = [{'ob_content_seg': ['我的','我的','阿噗'], 'ob_label':1}, {'ob_content_seg':['我的','我的','阿噗'], 'ob_label':0},
    #          {'ob_content_seg':['我','我他','阿噗大'], 'ob_label': 1}, {'ob_content_seg':['我的','我的哦哦','阿大噗'], 'ob_label': 0}]
    word_dict = {}
    # label_num={label1:num1, label2:num2, ...}
    label_num = {}
    for i in range(len(datas)):
        data = datas[i]
        if not label_num.get(data['ob_label']):
            label_num[data['ob_label']] = 1
        else:
            label_num[data['ob_label']] += 1
        ob_content_seg = data[key]
        if ob_content_seg:
            for word in set(ob_content_seg):
                if word.strip() != '':
                    if not word_dict.get(word.strip()):
                        word_dict[word.strip()] = {}
                        word_dict[word.strip()][data['ob_label']] = 1
                    else:
                        if not word_dict[word.strip()].get(data['ob_label']):
                            word_dict[word.strip()][data['ob_label']] = 1
                        else:
                            word_dict[word.strip()][data['ob_label']] += 1
    # for a in word_dict:
    #     print a, word_dict[a]

    print len(word_dict)
    print label_num
    label_list = label_num.keys()
    print label_list

    # word_dict={word1:{label1: , label2: ,...'chi': {label1: , label2: ,...}}, word2:{}...}
    for word in word_dict:
        for label in label_list:
            if not word_dict[word].get(label):
                word_dict[word][label] = 0
        # 计算词的CHI
        #
        word_dict[word]['chi'] = {}
        for label in label_list:
            a = word_dict[word][label]
            b = 0
            for i in label_list:
                if i != label:
                    b += word_dict[word][i]
            c = label_num[label] - a
            tmp = 0
            for i in label_num:
                if i != label:
                    tmp += label_num[i]

            d = tmp - b
            word_dict[word]['chi'][label] = float((a*c - b*d)**2) / float((a+b)*(c+d))

    word_list = word_dict.items()
    # 删掉停用词
    stop_words_list = utils.get_stop_words()
    i = 0
    while i < len(word_list):
        if len(word_list[i][0]) < 2 or word_list[i][0] in stop_words_list:
            del word_list[i]
        else:
            i += 1
    sort_lists = []
    for label in label_list:
        lst = sorted(word_list, key=lambda x: x[1]['chi'][label], reverse=True)
        words = [word[0] for word in lst]
        sort_lists.extend(words[:num])
    sort_lists = set(sort_lists)
    return list(sort_lists)

        #
    #     a = word_dict[word][0]
    #     b = word_dict[word][1]
    #     c = label_num[0] - word_dict[word][0]
    #     d = label_num[1] - word_dict[word][1]
    #     word_dict[word]['chi'] = float((a*c - b*d)**2) / float((a+b)*(c+d))
    # word_list = word_dict.items()
    # # 删掉停用词
    # stop_words_list = utils.get_stop_words()
    # i = 0
    # while i < len(word_list):
    #     if len(word_list[i][0]) < 2 or word_list[i][0] in stop_words_list:
    #         del word_list[i]
    #     else:
    #         i += 1
    #
    # word_list.sort(key=lambda x: x[1]['chi'], reverse=True)
    # words = [word[0] for word in word_list]
    #
    # return words[:num]
# for id, i in enumerate(word_list):
#     print id, i[0], i[1]
if __name__ == '__main__':
    # for id, i in enumerate(get_chi_wordlist(10, 'ob_content_seg')):
    #     print id
    #     for j in i:
    #         print j,
    for iden, i in enumerate(get_chi_wordlist(10, 'ob_content_seg')):
        print iden, i

