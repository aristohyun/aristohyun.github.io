---
layout: post
title: "기계학습, 선형 회귀 Linear Regression"
description: "Regression, 회귀분석"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Regression, R^2]
use_math: true
redirect_from:
  - /2021/06/25/
---

* Kramdown table of contents
{:toc .toc}

[집값 예측 : 회귀분석](https://www.kaggle.com/s1hyeon/house-price-regression/edit "캐글, House Price Predict"){: target="_blank"}    


# Regression, 회귀 분석    
> 회귀분석이란, 변수들 사이의 관계를 추정하는 분석방법으로    
> 주어진 데이터들을 가장 잘 설명하는 관계식(회귀식), 하나의 선(회귀선)을 구하는 방법이라고 할 수 있다


## 회귀 분석의 4단계
1. 이론의 가정    
2. 회귀직선 그리기 : 최소제곱법, $R^2$    
3. 가설검정 : 모수 추정 (실제 선형관계 추정)    
4. 평균값에 대한 예측, 개별값에 대한 예측    


### 이론의 가정    
1. 두 변수 간 선형관계가 있어야한다
2. 표본추출이 무작위하게 이루어져야 한다
3. X의 값이 두개 이상이여야 한다
4. Zero-conditional Mean : 오차들의 평균은 0이되어야 한다
5. 등분산성, Homoskedaticity : 오차들이 같은 정도로 퍼져있어야 한다
6. 독립성, Independence : 오차항들끼리 독립적이어야 한다
7. 정규성, Normalty : 오차들끼리는 정규분포를 이루어야 한다


### 회귀직선 그리기

![단순 선형회귀](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile7.uf.tistory.com%2Fimage%2F997E924F5CDBC1A6283C93)    
기본적으로 단순 선형 회귀식은 다음과 같다    
$Y = b_{0} + b_{1}X$       

독립변인이 늘어나면 다음과 같아진다 (다중 선형 회귀식)    
$Y = b_{0} + b_{1}X_{1} + b_{2}X_{2} + b_{3}X_{3} ...$    

### 결정계수, R-Squared, $R^2$
> 이때 내가 찾은 선형식이 이 데이터를 정말 잘 표현하는지 확인할 때 $R^2$를 계산한다    
> $R^2$는 독립변수가 종속변수를 얼마만큼 설명해주는지를 가리키는 지표이며    
> 0에 가까울수록 설명력이 낮으며, 1에 가까울수록 설명력이 높다    
  
**$R^2 = \frac{SSE}{SST} = 1 - \frac{SSR}{SST}$**    
  
SST = Total Sum of Squares, 관측값 - 평균값     
**$\sum\limits_{i=1}^{n}(y_i - \bar{y})^2$**    
    
SSE = Explained Sum of Squares, 추정값 - 평균값   
**$\sum\limits_{i=1}^{n}(\hat{y}_i - \bar{y})^2$**    
    
SSR = Residual Sum of Squares, 관측값 - 추정값(잔차의 합)    
**$\sum\limits_{i=1}^{n}(y_i - \hat{y})^2$**    
    
$y_i$ : 관측값
$\bar{y}$ : 관측값의 평균값
$\hat{y}$ : 추정값 (회귀식의 값)

#### SST = SSE + SSR 유도

$    
SST = \sum\limits_{i=1}^{n}(y_i - \bar{y})^2    
= \sum\limits_{i=1}^{n}(y_i - \hat{y}_i + \hat{y}_i - \bar{y})^2    
= \sum\limits_{i=1}^{n}((y_i - \hat{y}_i)^2 + 2((y_i - \hat{y}_i)(\hat{y}_i - \bar{y}) +(\hat{y}_i - \bar{y})^2)    
= \sum\limits_{i=1}^{n}(y_i - \hat{y}_i)^2 + \sum\limits_{i=1}^{n}2((y_i - \hat{y}_i)(\hat{y}_i - \bar{y}) + \sum\limits_{i=1}^{n}(\hat{y}_i - \bar{y})^2
= SSE + SSR + \sum\limits_{i=1}^{n}2((y_i - \hat{y}_i)(\hat{y}_i - \bar{y})    
$    
