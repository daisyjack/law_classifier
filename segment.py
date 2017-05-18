# coding: utf-8
import jieba
import cPickle as pickle


# 分词，然后把结果存入.pkl
file_name = '/home/ren/law_crawler/data_law_tf/data_law_tf2_.pkl'
in_file = open(file_name, 'rb')
contents = pickle.load(in_file)
in_file.close()
for i in range(len(contents)):
    content = contents[i]
    if content['ob_content']:
        ob_seg_list = jieba.cut(content['ob_content'])
        content['ob_content_seg'] = list(ob_seg_list)
    else:
        content['ob_content_seg'] = None
    if content['ob_comment_content']:
        ob_comment_seg_list = jieba.cut(content['ob_comment_content'])
        content['ob_comment_content_seg'] = list(ob_comment_seg_list)
    else:
        content['ob_comment_content_seg'] = None

out_file = open(file_name, 'wb')
pickle.dump(contents, out_file, True)
out_file.close()
in_file = open(file_name, 'rb')
contents = pickle.load(in_file)
# print contents[0]['ob_content_seg'].decode('utf-8'), contents[0]['ob_comment_conent_seg'].decode('utf-8')
in_file.close()
