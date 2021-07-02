---
layout: post
title: "기계학습, 정규화 회귀 Reqularization Regression"
description: "Regression, 회귀분석"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Regression, Lasso, Ridge]
use_math: true
redirect_from:
  - /2021/06/25/
---

* Kramdown table of contents
{:toc .toc}

[참고 사이트](https://rk1993.tistory.com/entry/Ridge-regression%EC%99%80-Lasso-regression-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0){: target="_ blank"}
[참고 사이트](https://modern-manual.tistory.com/21){: target="_ blank"}


# Reqularization
> 모델이 Overfitting 될 수 있다. 그렇기에 정규화를 수행해 일반화를 가능케 한다

기존 선형 회귀에서는 SSR이 최소가 되도록만 했는데,     
Ridge와 Lasso에서는 여기에 계수의 크기도 최소가 되게 하여 정확도를 높인다      


# Ridge Regression    
> L2 Regression    
> 잔자제곱합(SSR) + 패널티 항(베타 값)    
$\;\; \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ i)^2 + \lambda\sum\limits_ {j=1}^{m}(b_ j)^2$

$ \lambda $ : SSR과 패널티 항의 비중을 조절해 주기 위한 값
- $ \lambda = 0 $ : $ \lambda $ 가 0이면, 일반적인 선형 회귀 모형이랑 같아짐 
- $ \lambda > $ :  $ \lambda $ 가 클수록, 패널티 정도가 커지기에 가중치 값을 그만큼 작게 해야함. 그러나 너무 커지면 오히려 정확도가 떨어질 수 있음 

- 크기가 큰 변수를 우선적으로 줄이는 경향이 있음    
- 변수간 상관관계가 높은 상황에서 좋은 성능을 기대할 수 있음    

# Lasso Regression    
> L1 Regression    
> 잔자제곱합(SSR) + 패널티 절댓값 항(베타 값)    
$\;\; \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ i)^2 + \lambda\sum\limits_ {j=1}^{m} \mid b_ j \mid$

- 중요한 몇개의 변수만 선택하고, 다른 계수들을 0으로 줄임  

# 패널티 값을 효과적으로 구하려면?    
다양한 람다 값을 입력해서 검증실험을 해보는 것    
즉 교차검증 CV를 하는것

[RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html){: target="_ blank"}     
[LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html){: target="_ blank"}    

~~~ python    
from sklearn.linear_model import RidgeCV
alphas = [0.01, 0.05, 0.1, 0.2, 1.0, 10.0, 100.0, 1000.0]
ridge = RidgeCV(alphas=alphas, normalize=True, cv=3)

from sklearn.linear_model import LassoCV
reg = LassoCV(cv=5, random_state=0)
# alphas are set automatically
~~~    
