---
layout: post
title: "기계학습, Pre-Processing Data 2"
description: "Pre-Processing Data, 데이터 전처리"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Unsupervised Learning, Pre Processing Data, Regul, ]
use_math: true
redirect_from:
  - /2021/06/25/
---

* Kramdown table of contents
{:toc .toc}      


[pipeline](https://hhhh88.tistory.com/6)

# 표준화, Standardization

> 값의 범위(scale)를 평균 0, 분산 1이 되도록 변환          
> 머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지           
> 딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)               
> 정규분포를 표준정규분포로 변환하는 것과 같음              

## 표준 점수, Z-score    

$ z_ i = \frac{x_ i - mean(x)}{std(x)} $        

> 데이터를 표준 정규 분포(가우시안 분포)에 해당하도록 값을 바꿔주는 과정            
> -1 ~ 1 사이에 68%가 있고, -2 ~ 2 사이에 95%가 있고, -3 ~ 3 사이에 99%가 있음                      
> -3 ~ 3의 범위를 벗어나면 outlier일 확률이 높음                   



# 일반화(정규화), Normalization
               
> 머신러닝 모델은 데이터가 가진 feature(특징)을 뽑아서 학습하는데, 이때 모델이 받아들이는 데이터의 크기가 들쑥날쑥하다면               
> 모델이 데이터를 이상하게 해석할 우려가 있음               
> ex) 아파트 가격에서 연식, 가격, 방갯수를 feature로 받았을 때 각각이 2000, 2억, 5 라고 한다면, 2억만 중요하게 여길 수 있음               
> 
> 즉 일반화란, 데이터의 범위(단위)를 일정하게(0 ~ 1) 만들어 모든 데이터가 같은 정도의 스케일(중요도)로 반영되도록 해주는 것              
> min-max의 편차가 크거나 다른 열에 비해 데이터가 지나치게 큰 열에 사용    


## Min-Max Normalization (최소-최대 정규화)

$ x_ {new\_i} = \frac{x_ i - min(x)}{max(x) - min(x)}   $

모든 데이터 중에서 가장 작은 값을 0, 가장 큰 값을 1로 두고, 나머지 값들은 비율을 맞춰서 모두 0과 1 사이의 값으로 스케일링해주는 것             
그러나 이상치(outlier)에 대해 취약하다는 단점이 있음                 


# 정규화(규제), Regularization



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


[참고 사이트](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gdpresent&logNo=221730873049){:target="_ blank"}              
[참고 사이트2](https://inuplace.tistory.com/515){:target="_ blank"}            
[참고 사이트3](https://hhhh88.tistory.com/6){:target="_ blank"}            

~~~ python

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


~~~ python

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['carat', 'depth']),
        ('cat', categorical_transformer, ["cut"])])
        
        
clf = Pipeline(steps=[('preprocessor', preprocessor),("logistic", LogisticRegression())])
clf.fit(X_train.head(1000), y_train.head(1000),)

~~~
