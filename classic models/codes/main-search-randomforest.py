# Author: Wajahat Riaz 
# License: Apache-2.0
# Github Link: https://github.com/WajahatRiaz/SingleLabelTextClassification.git

from __future__ import unicode_literals, print_function, division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
# TfidfVectorizer是一种文本特征提取器，它可以将文本数据转换为基于词频-逆文档频率（TF-IDF）算法的特征向量。TF-IDF是一种常用于信息检索和文本挖掘的加权技术，用于评估一个单词对于一个文档集合中的某个文档的重要性。

# 具体来说，TF-IDF算法计算一个单词在一个文档中出现的频率（TF），并乘以一个与该单词在整个文档集合中出现的文档频率（IDF）相关的权重。这个权重可以用来衡量一个单词在整个文档集合中的重要性，因为如果一个单词在整个文档集合中都很常见，那么它在一个特定的文档中出现的可能性就不太能够区分这个文档和其他文档。

# TfidfVectorizer将每个文档转换为一个向量，其中每个元素表示一个单词的TF-IDF分数。这些向量可以用作机器学习算法的输入特征。它通常被用于文本分类、信息检索等任务中。
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# DecisionTreeClassifier 类定义了一个决策树分类器，它位于 sklearn.tree 包中。决策树是一种基于树形结构的监督学习算法，用于分类和回归问题。

# RandomForestClassifier 类定义了一个随机森林分类器，它位于 sklearn.ensemble 包中。随机森林是一种基于决策树的集成学习算法，它将多个决策树组合成一个更强大的模型。

# LinearSVC 类定义了一个线性支持向量机分类器，它位于 sklearn.svm 包中。支持向量机是一种基于间隔最大化的监督学习算法，用于分类和回归问题。
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

TOTAL_CLASSES = 4
ITERATIONS = 4 # Number of iterations to split, train and evaluate the data set
classes = ['student', 'course', 'faculty', 'project']
tags_of_classes = [1,2,3,4]

# 定义了一个名为classifiers的元组列表，其中每个元组都包含两个元素：一个字符串和一个机器学习分类器对象

filenames = ['..\dataset\webkb-train-stemmed.txt',
             '..\dataset\webkb-test-stemmed.txt']

with open('..\dataset\merge-stemmed.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
# 对于每个文件，代码块中的第二个with语句定义了一个名为infile的文件对象，用于打开当前文件并读取它的内容。然后，infile对象的read()方法被调用，以读取整个文件的内容并将其作为一个字符串返回。


file_contents = pd.read_csv("..\dataset\merge-stemmed.txt", header=None, sep='\t', 
                       names=['label', 'text'])

print(file_contents)


def count(data_frame):

    for i in range (TOTAL_CLASSES):
        print(f"Number of times {classes[i]} appeared", len(data_frame[data_frame.label==classes[i]]))
    
def tagging (data_frame):
    for i in range(TOTAL_CLASSES):
         data_frame.loc[data_frame["label"]==classes[i],"label"]=tags_of_classes[i]
    
    return data_frame

def to_tfidf(data_frame):


    df_y=data_frame["label"]
    df_x=data_frame["text"]

    tfidf = TfidfVectorizer(min_df=1, stop_words='english')
    df_x = tfidf.fit_transform(df_x.fillna('')).toarray()
    with open('feature_vectors.txt', 'w') as f:
        np.savetxt(f, df_x)

    return df_x, df_y
   
tagged_data_frame = tagging(file_contents)

print(tagged_data_frame)
x, y = to_tfidf(tagged_data_frame)
y_train = y.astype('int')
list_f1_scores = []
from sklearn.model_selection import GridSearchCV
import numpy as np

# 定义超参数的取值范围
param_grid = {
    # 'n_estimators': list(range(450,550,10)),
    #  'criterion': ['gini', 'entropy'],
    #   'max_depth': [None] + list(range(20,50, 5)),
    # 'min_samples_split': list(range(20,40,4)),
    'max_features': ['auto', 'sqrt', 'log2']
}

# 定义 RandomForestClassifier 模型
rf = RandomForestClassifier(random_state=42,class_weight='balanced',n_estimators=470,criterion='gini',max_depth=25,min_samples_leaf=2,min_samples_split=32,)

# 定义 RandomizedSearchCV 实例
grid_search = GridSearchCV(rf, param_grid, cv=5,scoring='f1_weighted',verbose=2,n_jobs=-1)
grid_search.fit(x, y_train)

# 输出最佳模型的超参数和得分
print('Best hyperparameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)




