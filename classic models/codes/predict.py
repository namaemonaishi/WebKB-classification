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
custom_colors = ['lightgray']  # 替换为你想要的较淡的颜色名称或十六进制代码
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

TOTAL_CLASSES = 4
ITERATIONS = 5 # Number of iterations to split, train and evaluate the data set
classes = ['student', 'course', 'faculty', 'project']
tags_of_classes = [1,2,3,4]

classifiers = (("SVM (Linear)", LinearSVC(class_weight='balanced',penalty='l2', loss='hinge', C=1)),
               ("Decision Tree", DecisionTreeClassifier(class_weight='balanced',max_depth=14,criterion="gini",min_samples_leaf=14,min_samples_split=2,max_features="sqrt")),
               ("Random Forest", RandomForestClassifier(class_weight='balanced',n_estimators=470,criterion='gini',max_depth=25,min_samples_leaf=2,min_samples_split=32)))
# 定义了一个名为classifiers的元组列表，其中每个元组都包含两个元素：一个字符串和一个机器学习分类器对象

file_contents = pd.read_csv("..\webkb\webkb.txt", header=None, sep='\t', 
                       names=['label', 'text'])
    
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
   
def apply_models(model, data_frame, iteration):
    """Returns a list of (model_name, accuracy) that results from 
    applying each model to the train and test dataframes.
    use_lsi indicates if it should be applied to the documents features
    before using the model; default False."""

    tagged_data_frame = tagging(data_frame)

    print(tagged_data_frame)
    x, y = to_tfidf(tagged_data_frame)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    list_f1_scores = []

    for (name,model) in classifiers:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        list_f1_scores.insert(i,f1_score(y_test, y_pred, average='weighted'))
        print(f"Iteration %d:Classification report for {name}:\n"
        f"{classification_report(y_test, y_pred)}\n" %(iteration+1))
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.figure_.suptitle(f"Iteration %d: Confusion matrix {name}"%(iteration+1))
        print(f"Iteration %d: Confusion matrix for the {name}:\n{disp.confusion_matrix}" %(iteration+1))

        Model_report = classification_report(y_test, y_pred, output_dict=True)
        model_df = pd.DataFrame(Model_report).transpose()
        plt.xticks(range(TOTAL_CLASSES),classes)
        plt.yticks(range(TOTAL_CLASSES),classes)
        plt.savefig(f"..\\pics\\{name}_%d.jpg" %(iteration+1))
   
    return np.array(list_f1_scores)

f1_score_matrix = np.eye(ITERATIONS, 3)

for i in range(ITERATIONS):
    f1_score_matrix[i,:] = apply_models(classifiers, file_contents, i)
    
weighted_f1_scores = [mean(f1_score_matrix[:,i]) for i in range(3)]

print(weighted_f1_scores)

def t_test_paired(x1,x2):

    print("Running t-test...\n")
    x1_bar, x2_bar = np.mean(x1), np.mean(x2)
    n1, n2 = len(x1), len(x2)
    var_x1, var_x2= np.var(x1, ddof=1), np.var(x2, ddof=1)
    # 这段代码的作用是计算两个一维数组 x1 和 x2 的样本方差，其中 var_x1 和 var_x2 分别表示 x1 和 x2 的样本方差。
    # 具体来说，np.var() 是 NumPy 库中计算方差的函数，其中 ddof 参数表示自由度的校正值，通常设置为 1，表示对样本方差进行无偏估计。
    var = ( ((n1-1)*var_x1) + ((n2-1)*var_x2) ) / (n1+n2-2)
    std_error = np.sqrt(var * (1.0 / n1 + 1.0 / n2))
  
    print("x1:",np.round(x1_bar,4))
    print("x2:",np.round(x2_bar,4))
    print("variance of first sample:",np.round(var_x1))
    print("variance of second sample:",np.round(var_x2,4))
    print("pooled sample variance:",var)
    print("standard error:",std_error)
    # std_error 是进行学生 t 检验时需要用到的标准误差，它是对样本均值的估计值与真实总体均值之间的差异进行测量的一个指标。

    # calculate t statistics
    t = abs(x1_bar - x2_bar) / std_error
    print('t static:',t)
    # two-tailed critical value at alpha = 0.05
    t_c = stats.t.ppf(q=0.975, df=17)
    print("Critical value for t two tailed:",t_c)
 
    # one-tailed critical value at alpha = 0.05
    t_c = stats.t.ppf(q=0.95, df=12)
    print("Critical value for t one tailed:",t_c)
 
    # get two-tailed p value
    p_two = 2*(1-stats.t.cdf(x=t, df=12))
    print("p-value for two tailed:",p_two)
  
    # get one-tailed p value
    p_one = 1-stats.t.cdf(x=t, df=12)
    print("p-value for one tailed:",p_one)

t_test_paired(f1_score_matrix[:,0],f1_score_matrix[:,1])
t_test_paired(f1_score_matrix[:,0],f1_score_matrix[:,2])
t_test_paired(f1_score_matrix[:,1],f1_score_matrix[:,2])