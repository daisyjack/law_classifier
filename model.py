# coding: utf-8
import cPickle as pickle
import chi
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

# 选法院认为还是事实
key = 'ob_comment_content_seg'
k_scores = []
feature_num_range = range(100)
feature_num = 15
# for feature_num in feature_num_range:
#     feature_num += 1
file_name_all = '/home/ren/law_crawler/data_law_tf/data_law_tf_all.pkl'
in_file = open(file_name_all, 'rb')
# datas=[{'url','content','title','ob_label','ob_content','ob_comment_content','ob_content_seg','ob_comment_content_seg'},{},{},{}....]
datas = pickle.load(in_file)
in_file.close()
feature_words = chi.get_chi_wordlist(feature_num, key)
# 用CHI高的词在每个文档中的词频来表示每个文档
datas_feature = []
labels = []
for data in datas:
    data_feature = []
    for feature_word in feature_words:
        data_feature.append(data[key].count(feature_word) if data[key] else 0)
    datas_feature.append(data_feature)
    labels.append(data['ob_label'])
# print datas_feature
# print labels

# 将词频转换成tf*idf
transformer = TfidfTransformer()
tfidf_data = transformer.fit_transform(datas_feature)

# 分测试集和训练集
# X_train, X_test, y_train, y_test = train_test_split(tfidf_data, labels, test_size=0.3)
model = SVC()
# model.fit(X_train, y_train)
scores = cross_val_score(model, tfidf_data, labels, cv=5, scoring='recall')
print scores.mean()
k_scores.append(scores.mean())

# 学习曲线
# train_sizes, train_loss, test_loss = learning_curve(
#     model, tfidf_data, labels, cv=10, scoring='mean_squared_error',
#     train_sizes=[0.1, 0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
# plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
#          label="Training")
# plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
#         label="Cross-validation")
#
# plt.xlabel("Training examples")
# plt.ylabel("Loss")
# plt.legend(loc="best")
# plt.show()

# plt.plot(feature_num_range, k_scores)
# plt.xlabel('Value of feature_num')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()
# print 'score ------'
# print model.score(X_test, y_test)
