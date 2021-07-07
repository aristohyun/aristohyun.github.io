---
layout: post
title: "기계학습, Logistic Regression"
description: "로지스틱 회귀, Logistic Regression"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Logistic Regression, Regression]
use_math: true
redirect_from:
  - /2021/07/01/
---

* Kramdown table of contents
{:toc .toc}


[Titinic : logistic Regression](https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python){: target="_ blank"}


# Logistic Regression     
> 분류(Classification) 기법    
> 일반적인 회귀 분석의 목표와 동일하게 종속 변수와 독립 변수간의 관계를     
> 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것

## Odds, 승산    
> $ Odds = \frac{P}{1-P} $    
> 범주 0에 속할 확률 대비 범주 1에 속할 확률, 즉 두 확률의 비율     
> P는 범주 1에 속할 확률    

확률 P와 Odds값은 거의 유사     
case1. P( O | O+X ) = 1/2000 ≒ Odds( O | X )= 1/1999     
case2. P( O | O+X ) = 1/8000 ≒ Odds( O | X )= 1/7999  
<br />
Odds를 Odds로 나누어 비교 가능     
case2 대비 case1의 확률이 약 4배(≒4.0015) 더 높음      

## 로지스틱 함수    
>  $ y = \frac{1}{1 + e^{-(b_ 0 + b_ 1 x)}} = \frac{e^{b_ 0 + b_ 1 x}}{1 + e^{b_ 0 + b_ 1 x}} $    

![image](https://user-images.githubusercontent.com/32366711/124351211-fa8e3a80-dc33-11eb-8652-a461b344542a.png)

만약, 결과값이 사망/생존과 같이 이분법적인 방법으로 구분되는 경우    
선형회귀등으로 하면 적합하지 못한 모델이 됨    
왜?    
1. 선형 회귀분석의 식은 $ y = b_ 0 + b_ 1 x $, 이때 좌항은 0~1의 범위를 갖지만, 우항은 $-\infty ~ \infty$ 사이의 범위를 가지기에 올바르지 못함
2. X값이 1만큼 증가할 때 y값이 얼만큼 증가하는지 설명하기 어려움    

그렇기에 로짓변환을 통해 로지스틱 회귀 사용     

### Logit Transform, 로짓 변환    
P = 어떤 사건이 일어날 확률    
$ P = \pi(X = x) = \frac{1}{1 + e^{-(b_ 0 + b_ 1 x)}} $    

$ Odds = \frac{P}{1-P} $    
$ ln(Odds) = ln(\frac{\pi(X = x)}{1 - \pi(X = x)}) = ln(\frac{\frac{1} {1 + e^{-(b_ 0 + b_ 1 x)}} } {1 - \frac{1}{1 + e^{-(b_ 0 + b_ 1 x)}}}) $    
$ = b_ 0 + b_ 1 x $    
$ \therefore \; ln(Odds) = b_ 0 + b_ 1 x $     

따라서 이제 일반 회귀식으로 로짓값을 예측할 수 있음      

## 코스트 함수    

$ cost(W) = \frac{1}{m} \sum c(H(x), y) $
$ c(H(x), y) = 
\left\{\begin{matrix}
-log(H(x)) \;\;\;\; : y=1
\\ 
-log(1-H(x)) \;\; : y=0
\end{matrix}\right.
$


# 참고 사이트
[KorShort](https://blog.naver.com/tjqdl2013/220835520162){: target="_ blank"}
[Classic!](https://icefree.tistory.com/entry/%EA%B8%B0%EC%B4%88-%ED%86%B5%EA%B3%84-Odds-Ratio){: target="_ blank"}
