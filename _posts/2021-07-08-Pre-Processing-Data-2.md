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

$ $ x_ {new\_i} = \frac{x_ i - min(x)}{max(x) - min(x)} $   $

모든 데이터 중에서 가장 작은 값을 0, 가장 큰 값을 1로 두고, 나머지 값들은 비율을 맞춰서 모두 0과 1 사이의 값으로 스케일링해주는 것             
그러나 이상치(outlier)에 대해 취약하다는 단점이 있음                 


# 정규화(규제), Regularization



# 코드

## Pipeline

~~~ python

# 1. 데이터 수집, 로드
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC(gamma='auto')) ])

pipline.fit(X_train, y_train)

print('테스트점수 :{:.2f}'.format(pipline.score(X_test, y_test)))

pipline.get_params

~~~
