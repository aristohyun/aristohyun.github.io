---
layout: post
title: "기계학습, Pre-Processing Data 2"
description: "Pre-Processing Data, 데이터 전처리"
categories: [MachineLearning]
tags: [Machine Learning, Pre Processing Data]
use_math: true
redirect_from:
  - /2021/06/25/
---

* Kramdown table of contents
{:toc .toc}      


# 표준화, Standardization

> 주어진 데이터를 표준정규분포로 변환하는 것
> 값의 범위(scale)를 평균 0, 분산 1이 되도록 변환          
> 머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지           
> 딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)                        

## 표준 점수, Z-score    

@ z_ i = \frac{x_ i - mean(x)}{std(x)} @        

> 데이터를 표준 정규 분포(가우시안 분포)에 해당하도록 값을 바꿔주는 과정            
> -1 ~ 1 사이에 68%가 있고, -2 ~ 2 사이에 95%가 있고, -3 ~ 3 사이에 99%가 있음                      
> -3 ~ 3의 범위를 벗어나면 outlier일 확률이 높음                   

![image](https://user-images.githubusercontent.com/32366711/126485079-529d8966-282b-4294-ab67-230f612eaf44.png)


# 일반화(정규화), Normalization
               
> 머신러닝 모델은 데이터가 가진 feature(특징)을 뽑아서 학습하는데, 이때 모델이 받아들이는 데이터의 크기가 들쑥날쑥하다면               
> 모델이 데이터를 이상하게 해석할 우려가 있음               
> ex) 아파트 가격에서 연식, 가격, 방갯수를 feature로 받았을 때 각각이 2000, 2억, 5 라고 한다면, 2억만 중요하게 여길 수 있음               
> 
> 즉 일반화란, 데이터의 범위(단위)를 일정하게(0 ~ 1) 만들어 모든 데이터가 같은 정도의 스케일(중요도)로 반영되도록 해주는 것              
> min-max의 편차가 크거나 다른 열에 비해 데이터가 지나치게 큰 열에 사용    


## Min-Max Normalization (최소-최대 정규화)

@ x_ {new\_i} = \frac{x_ i - min(x)}{max(x) - min(x)}   @

모든 데이터 중에서 가장 작은 값을 0, 가장 큰 값을 1로 두고,        
나머지 값들은 비율을 맞춰서 모두 0과 1 사이의 값으로 스케일링해주는 것              
그러나 이상치(outlier)에 대해 취약하다는 단점이 있음                 


# 전처리 주의 사항

> 훈련 데이터셋을 그냥 전처리를 다해버리면 교차검증을 하려고 할 때, 테스트셋이 훈련셋과 동일하게 전처리가 되어있다면           
> 이건 그냥 자기 자신을 확인하는 것 밖에는 되지 않아서 올바른 예측을 할 수가 없음

예를 들어 2022의 정보를 알고싶지만, 아직 오 지않아 데이터가 없기에       
지금까지의 데이터들로 가지고 예측 모델을 만들어 2022을 예측하는 것이 예측 모델을 만드는 것의 목표       

그렇다면 2010년도의 정보로 2011을 잘 예측하는지 보고 2011까지의 정보로 2012를 잘 예측하는지             
2019까지의 정보로 2020을 잘 예측하는지 이렇게 확인을 해야하는데               
그냥 2020까지의 정보로 2015를 예측하는지 보는 것은 자기 자신을 확인하는 것 밖에는 안되기에                
통째로 전처리를 한 후에 교차 검증을 하는 것은 잘못됬다고 볼 수 있음              

그렇다면 어떻게 해야하는가?        
매번 교차검증을 할 때 데이터셋을 임의로 분리를 해서 따로 따로 전처리를 하고 확인하고 해야하는가?            

## Pipeline

> 위와 같은 상황을 막기 위해 사용        
> 파이프라인에 전처리 클래스를 넣어주면, 각 데이터들을 일시적으로 전처리 후 학습을 진행함           
> fit - transform - fit - transform - modeling 의 모든 과정을 하는 클래스           
> 해당 학습을 시킬 때에는 전처리가 되어 있지만, 기존 데이터에는 영향을 주지 않음          

> GridSearch로 교차검증을 할 때에도 이 파이프라인을 넣으면, 알아서 임의로 나눈 테스트셋과 훈련셋을 파이프라인을 통해서 결과를 내게됨



[pipeline](https://hhhh88.tistory.com/6)

[참고 사이트](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gdpresent&logNo=221730873049){:target="_ blank"}              
[참고 사이트2](https://inuplace.tistory.com/515){:target="_ blank"}            
[참고 사이트3](https://hhhh88.tistory.com/6){:target="_ blank"}            
[4](http://blog.skby.net/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8-machine-learning-pipeline/){:target="_ blank"}          
~~~ python
# svm, suport vector machine         
# 두 분류 사이의 여백을 최대화 하여 일반화 능력을 극대화 하는 모델      
# 여기서 여백이란, 분류선(Decision boundary)과 가장 가까운 데이터들(suport vector)간의 거리    

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)


# 표준화
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# 학습
svc = SVC(gamma="auto")
svc.fit(X_train, y_train)

# 예측 및 평가
pred = svc.predict(X_test)
print('테스트점수 :{:.2f}'.format(svc.score(X_test, y_test)))

~~~

~~~ python

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC(gamma='auto')) ])

pipline.fit(X_train, y_train)

print('테스트점수 :{:.2f}'.format(pipline.score(X_test, y_test)))

~~~
    
### Titanic data Practice

~~~ python
class Pclass(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, before):
        X = before.copy()
        pclass_train_dummies = pd.get_dummies(X['Pclass'])

        X.drop(['Pclass'], axis=1, inplace=True)

        X = X.join(pclass_train_dummies)
    
        return X
~~~

~~~ python
class Sex(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,before):
        X = before.copy()
        sex_train_dummies = pd.get_dummies(X['Sex'])
        sex_train_dummies.columns = ['Female', 'Male']
        X.drop(['Sex'], axis=1, inplace=True)
        X = X.join(sex_train_dummies)
        return X
~~~

~~~ python
class Age(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,before):
        X = before.copy()
        X["Age"].fillna(X["Age"].mean() , inplace=True)
        return X
~~~

~~~ python
class Fare(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,before):
        X = before.copy()
        X["Fare"].fillna(0, inplace=True)
        return X
~~~

~~~ python
class Cabin(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,before):
        X = before.copy()
        X = X.drop(['Cabin'], axis=1)
        return X
~~~

~~~ python
class Embarked(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,before):
        X = before.copy()
        X["Embarked"].fillna('S', inplace=True)
        embarked_train_dummies = pd.get_dummies(X['Embarked'])
        embarked_train_dummies.columns = ['S', 'C', 'Q']
        X.drop(['Embarked'], axis=1, inplace = True)
        X = X.join(embarked_train_dummies)
        return X
~~~

~~~ python
knnpipe = Pipeline([
    ('pclass', Pclass()),
    ('sex', Sex()),
    ('age', Age()),
    ('fare', Fare()),
    ('cabin', Cabin()),
    ('embarked', Embarked()),
    ('knn', KNeighborsClassifier())
])

logisticpipe = Pipeline(steps=[
    ('pclass', Pclass()),
    ('sex', Sex()),
    ('age', Age()),
    ('fare', Fare()),
    ('cabin', Cabin()),
    ('embarked', Embarked()),
    ('logistic', LogisticRegression())
])
~~~

~~~ python
from sklearn.model_selection import GridSearchCV

knn_param = {'knn__n_neighbors': np.arange(1,30)}
knn_cv = GridSearchCV(knnpipe, knn_param, cv=3).fit(x_train,y_train)# Fit  

logis_param = {'logistic__C': np.logspace(-3, 3, 7), 'logistic__penalty': ['l1', 'l2']}
logis_cv = GridSearchCV(logisticpipe, logis_param, cv=3).fit(x_train, y_train)

print(knn_cv.best_score_)
print(knn_cv.best_params_)
print(knn_cv.score(x_train, y_train))
print(knn_cv.score(x_test, y_test))

print(logis_cv.best_score_)
print(logis_cv.best_params_)
print(logis_cv.score(x_train, y_train))
print(logis_cv.score(x_test, y_test))

~~~

![image](https://user-images.githubusercontent.com/32366711/126462405-45b3e5b1-c27c-49fd-ba54-4f18784d7adc.png)

![image](https://user-images.githubusercontent.com/32366711/126462378-b814558e-1618-4b07-8d63-31d1480332f8.png)

![image](https://user-images.githubusercontent.com/32366711/126462453-bf1a6933-c895-4518-af50-bdf602843f2b.png)

