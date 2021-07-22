---
layout: post
title: "기계학습, Linear Regression 2"
description: "Linear Regression, 확률론적 선형 회귀"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Linear Regression, Regression]
use_math: true
redirect_from:
  - /2021/06/25/
---

* Kramdown table of contents
{:toc .toc}      

[통계적](https://velog.io/@jeromecheon/ML-4%ED%8E%B8-Linear-tasks-Regression%ED%9A%8C%EA%B7%80-1.%EC%B5%9C%EB%8C%80%EA%B0%80%EB%8A%A5%EB%8F%84-%EC%B5%9C%EC%86%8C%EC%A0%9C%EA%B3%B1%EB%B2%95){:target="_ blank"}

> 데이터가 확률 변수로부터 생성된 표본이라고 가정

# 확률론적 선형 회귀모형

## 가정

### 선형 정규 분포 가정

@
y \sim  N(w^Tx,\sigma^2 ) \\\ 
\epsilon = y - w^Tx \\\ 
p(\epsilon | x) = N(0,\sigma^2)
@

> 종속 변수 y가 독립변수 x의 선형조합으로 결정되는 기대값과 고정된 분산을 가지는 정규분포이다

### 외생성 가정

@
E \left [ \epsilon | x \right ] = 0
@




> 잡음의 기대값은 독립변수 x의 크기에 상관없이 항상 0이다


### 조건부 독립 가정

@
Cov \left [ \epsilon_ i ,  \epsilon_ j \right ] = 0 \\\ 
E \left [ \epsilon_ i ,  \epsilon_ j \right ] = 0
@

> i번째 표본의 잡음과 j번째 표본의 잡음의 공분산 값은 x에 상관없이 항상 0이다
> 즉 서로 독립이다

### 등분산성 가정

@
Cov \left [ \epsilon \right ] = E \left [ \epsilon \epsilon^T \right ] = \sigma^2 I
@

> 잡음들의 분산 값은 표본과 상관 없이 항상 같다


## 최대 가능도 방법

> 최대가능도방법 (maximum likelihood method) 또는 최대우도법은 어떤 확률변수에서 표집한 값들을 토대로 그 확률변수의 모수를 구하는 방법        
> 어떤 모수가 주어졌을 때, 원하는 값들이 나올 가능도를 최대로 만드는 모수를 선택하는 방법이며, 점추정 방식에 속한다         

![image](https://user-images.githubusercontent.com/32366711/126483109-399b8761-b06b-46c4-af2f-84047e62e1ba.png)

![image](https://user-images.githubusercontent.com/32366711/126483121-c7bd85de-786a-4d40-9901-4c185b34031b.png)

![image](https://user-images.githubusercontent.com/32366711/126483128-963ef02e-e6ec-4bbd-abce-e0e63a61f268.png)

![image](https://user-images.githubusercontent.com/32366711/126483151-2f810a59-ee91-432d-929f-bf999cbaf2d6.png)

![image](https://user-images.githubusercontent.com/32366711/126483163-ee963d92-fb87-4ca8-b76a-7615cfe33620.png)

