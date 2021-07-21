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


