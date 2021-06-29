---
layout: post
title: "기계학습, 회귀 분석"
description: "회귀분석과 교차검증"
categories: [Machine Learning]
tags: [Supervised Learning, Python, Regression, R^2]
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

> 1~4이 만족되면 최소제곱법을 통한 회귀분석은 일돤되고 편향되지 않은 결과를 도출하며,   > 5~6이 만조되면 최소제곱법을 통한 회귀분석이 최선의 방법이며,    
> 7이 만족되지 않을 경우 표본의 관찰 값을 늘려서 중심극한 정리를 통해 정규성을 근사적으로 만족시킬 수 있다    

### 회귀직선 그리기

기본적으로 선형 회귀식은 다음과 같다
`$\hat{y}_{i} = b_{0} + b_{1}x_{i}$


### 가설검정

### 평균값에 대한 예측

