# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:52:10 2019

@author: Hosik
"""

#*************************************************
#실습1: 단순 베이즈 분류 
#*************************************************
#민주당 또는 공화당의 값을 갖는 분류 변수와 16개의 이진변수로 구성된
#HouseVotes84자료에 대하여 단순 베이즈 분류를 적용하여 보자.

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix 

# 자료불러오기 
df = pd.read_csv('./datasets/HouseVotes84.csv')
# na값 제거
df = df.dropna()

y = df['Class']
X = df.drop('Class', axis = 1)
for col in X.columns:  # Iterate over chosen columns
    #X1[col] = X1[col].apply(lambda x: 1 if x== 'y' else 0)
    X.loc[X[col] == 'y' , col] = 1
    X.loc[X[col] == 'n' , col] = 0

# alpha: 라플라스 수정 옵션 
bnb = BernoulliNB(alpha=0).fit(X, y)
#left: democrat, right: republican
print(np.round(bnb.predict_proba(X[0:10])), 5)
# 오분류표 행렬
print(confusion_matrix(y, bnb.predict(X)))

#*************************************************
#실습2: k-nn
#*************************************************
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
tr_iris, ts_iris, tr_target, ts_target = \
    train_test_split(iris.data, iris.target, test_size=0.5, 
    random_state = 1)
clf = KNeighborsClassifier(n_neighbors=3).fit(tr_iris, tr_target)
print("error: {:.3f}".format(1-clf.score(ts_iris, ts_target)))

#*************************************************
#실습3: Logistic Neuron 구현
#*************************************************

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.data.shape, cancer.target.shape)
# 박스그림 
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()

#target 분포확인
np.unique(cancer.target, return_counts=True)

x = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split
#평가용 자료 20% 
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, stratify=y, 
    test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)
#비율 확인
np.unique(y_train, return_counts=True)

# [DO19, p.99]
class LogisticNeuron:
    def __init__(self):
        self.w = None
        self.b = None

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식 계산
        return z

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트 계산
        b_grad = 1 * err    # 절편에 대한 그래디언트 계산
        return w_grad, b_grad

    def activation(self, z):
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a
        
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])      # 가중치 초기화
        self.b = 0                        # 절편 초기화
        for i in range(epochs):           # epochs만큼 반복
            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복
                z = self.forpass(x_i)     # 정방향 계산
                a = self.activation(z)    # 활성화 함수 적용
                err = -(y_i - a)          # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w -= w_grad          # 가중치 업데이트
                self.b -= b_grad          # 절편 업데이트
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]    # 정방향 계산
        a = self.activation(np.array(z))        # 활성화 함수 적용
        return a > 0.5

neuron = LogisticNeuron()
neuron.fit(x_train, y_train)
np.mean(neuron.predict(x_test) == y_test)

#*************************************************
#실습4: Logistic Regression 해석
#*************************************************
# statsmodels.api 활용
import pandas as pd
import numpy as np
import statsmodels.api as sm

nodal = pd.read_csv("./datasets/nodal.csv", engine='python')
rd = nodal.drop("m", axis=1)
Y = rd["r"]
X = rd.drop("r", axis=1)
X = sm.add_constant(X)
logit = sm.Logit(Y,X)
logit_res = logit.fit()
print(logit_res.summary())

# sklearn.linear_model.LogisticRegression 활용
# sklearn.linear_model.SGDClassifier 활용

#*************************************************
#실습5: Sonar자료
#*************************************************
# sklearn의 linear_model 모듈이용
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

## Read data
sonar = pd.read_csv("./datasets/sonar.csv", engine='python')
sonar['Class'].value_counts()
sonar.loc[sonar['Class']== 'R'] = 0
sonar.loc[sonar['Class']== 'M'] = 1
#sonar["Class"] = np.where(sonar['Class'] == 'R', 0, 1)

## Plot the correlation matrix
corr0 = sonar.loc[sonar['Class']==0].drop(['Class'],axis=1).corr()
corr1 = sonar.loc[sonar['Class']==1].drop(['Class'],axis=1).corr()

fig = plt.figure(figsize=(7, 5))
plt.subplot(1, 2, 1)
plt.pcolormesh(corr0)
plt.title("Rock")
plt.subplot(1, 2, 2)
plt.pcolormesh(corr1)
plt.title("Metal")
plt.show()

# 목표변수 구성하기
y = sonar['Class'].values
x = sonar.drop('Class', axis=1).values 
type(x)
#pandas.core.frame.DataFrame
#x = sonar.drop('Class', axis=1).values
#type(x)
#numpy.ndarray

# 데이터 분할
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=1)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 로지스틱함수 적합
logit_model = LogisticRegression(random_state=123)
logit_model.fit(x_train, y_train)

# 적합 회귀계수 
logit_model.coef_
logit_model.intercept_

# 예측
logit_model.predict(x_test)
logit_model.predict_proba(x_test)
sonar['Class'].value_counts()
pred = logit_model.predict(x_test)
#pred = np.where(pred == 'R', 0, 1)
pd.Series(pred).value_counts()
score = logit_model.score(x_test, y_test)  # 정분류율 반환
#
# prob = logit_model.predict_proba(x_test)[:,1]

# confusion matrix(오분류표)
metrics.confusion_matrix(y_test, pred)
#array([[31, 0],
#       [ 0, 32]], dtype=int64)

# ROC 그래프 그리기
# 모델에 의한 예측 확률 계산
y_pred_proba = logit_model.predict_proba(x_test)[::,1]

# fpr: 1-특이도, tpr: 민감도, auc 계산
fpr, tpr, _ = metrics.roc_curve(y_true=y_test, y_score=y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC 그래프 생성
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="로지스틱 회귀\n곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
# plt.axhline(y=0.7, color='red', label='민감도 기준선')
# plt.axvline(x=0.2, color='green', label='1-특이도 기준선')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + './result/logistic_ROC.png')
plt.show()




























