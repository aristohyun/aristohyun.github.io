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

[참고 사이트](https://rk1993.tistory.com/entry/Ridge-regression%EC%99%80-Lasso-regression-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0){: target="_blank"}
[참고 사이트](https://modern-manual.tistory.com/21){: target="_blank"}


# Reqularization
> 모델이 Overfitting 될 수 있다. 그렇기에 정규화를 수행해 일반화를 가능케 한다

기존 선형 회귀에서는 $ R^2 $가 최소가 되도록만 했는데,     
Ridge와 Lasso에서는 여기에 계수의 크기도 최소가 되게 하여 정확도를 높인다      


# Ridge Regression    
> L2 Regression    
> 잔자제곱합(SSR) + 패널티 항(베타 값)    


# Lasso Regression    
> L1 Regression    
> 잔자제곱합(SSR) + 패널티 절댓값 항(베타 값)    


