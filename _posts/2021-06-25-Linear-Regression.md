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

@ R^2 = \frac{SSE}{SST} = 1 - \frac{SSR}{SST} @    

> 이때 내가 찾은 선형식이 이 데이터를 정말 잘 표현하는지 확인할 때 $R^2$를 계산한다    
> $R^2$는 독립변수가 종속변수를 얼마만큼 설명해주는지를 가리키는 지표이며    
> 0에 가까울수록 설명력이 낮으며, 1에 가까울수록 설명력이 높다       
     
     
SST = Total Sum of Squares, 관측값 - 평균값     
$ = \sum\limits_{i=1}^{n}(y_ i - \bar{y})^2$    
    
SSE = Explained Sum of Squares, 추정값 - 평균값   
$ = \sum\limits_{i=1}^{n}(\hat{y}_ i - \bar{y})^2$    
    
SSR = Residual Sum of Squares, 관측값 - 추정값(잔차의 합)    
$ = \sum\limits_{i=1}^{n}(y_ i - \hat{y}_ i)^2$    
    
$y_ i$ : 관측값  
$\bar{y}$ : 관측값의 평균값  
$\hat{y}_ i$ : 추정값 (회귀식의 값)  

#### SST = SSE + SSR 유도    
[참고](https://datalabbit.tistory.com/51){:target="_ blank"}
회귀분석에서는 SSR ( Residual Sum of Square )이 최소가 되도록 해야함    

@ \hat{y} = b_0 + b_1 x \\\ 
SSR = \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ i)^2 = \sum\limits_ {i=1}^{n}(y_ i - b_0 + b_1 x_ i)^2 @    

$ \;\;\;\; b_ 0, b_ 1 $에 대하여 미분했을 때 0이 되야 SSR이 최소값 [^1]   
@ \begin{align\*}
1) \;\; \frac{\partial }{\partial b_ 0} \sum\limits_ {i=1}^{n}(y_ i - b_0 + b_1 x_ i)^2 &= -2 \sum\limits_ {i=1}^{n} (y_ i - b_0 - b_1 x_ i) \\\ 
&= -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 \end{align\*} @ 
  
@ \begin{align\*}
2) \;\; \frac{\partial }{\partial b_ 1} \sum\limits_ {i=1}^{n}(y_ i - b_0 + b_1 x_ i)^2 &= -2 \sum\limits_ {i=1}^{n} (y_ i - b_0 - b_1 x_ i) x_ i  \\\ 
&= -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i = 0  \end{align\*} @     

@1) \; \therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 \\\ 
2) \; \therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i = 0 @

@ \begin{align\*}
SST &= \sum\limits_{i=1}^{n}(y_ i - \bar{y})^{2} = \sum\limits_{i=1}^{n}(y_ i - \hat{y}_ {i} + \hat{y}_ i - \bar{y})^2 \\\ 
&= \sum\limits_{i=1}^{n}((y_ i - \hat{y}_ {i})^2 + 2(y_ i - \hat{y}_ {i})(\hat{y}_ {i} - \bar{y}) +(\hat{y}_ {i} - \bar{y})^2) \\\ 
&= \sum\limits_{i=1}^{n}(y_ i - \hat{y}_ {i})^2  +  \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ {i} - \bar{y})  +  \sum\limits_ {i=1}^{n}(\hat{y}_ {i} - \bar{y})^2 \\\ 
&= SSR + SSE + \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) 
\end{align\*}@

@ \begin{align\*}
\sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) &= \sum\limits_{i=1}^{n}2((y_ i - \hat{y}_ i) \hat{y}_ i - (y_ i - \hat{y}_ i) \bar{y}) \\\ 
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)\hat{y}_ i - 2 \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \bar{y} \\\ 
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(b_0 + b_1 x_ i) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + 2 b_1 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= \; 0 \;\;\; \because \;  1) [^2] , 2) [^3] \end{align\*}@

@\therefore \;\; \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) = 0 @

@\therefore \;\; SST = SSE + SSR @


#### SST = SSR + SSE 일반화 
@ \hat{y} = b_ 0 + b_ 1 x_ 1 + b_ 2 x_ 2 \cdots b_ n x_ n \\\
= b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ j \\\
SSR = \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ i)^2 = \sum\limits_ {i=1}^{n}(y_ i - b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ {ji} )^2 @

$\;\;\;\; b_ i $에 대하여 각각 미분했을 때 0이 되야 SSR이 최소값을 가짐 [^1]     
 
@ 1) \; \frac{\partial }{\partial b_ 0} \; \sum\limits_ {i=1}^{n}(y_ i - b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ {ji})^2 = \\\
-2 \sum\limits_ {i=1}^{n} (y_ i - b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ {ji})  = -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 \\\
\therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 @

@ 2)  \; \frac{\partial }{\partial b_ k} \; \sum\limits_ {i=1}^{n}(y_ i - b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ {ji})^2 = \;\;\;\;\;\; (1\leq k\leq n \;,\;\; k\in \mathbb{\mathbb{N}}) \\\
 -2 \sum\limits_ {i=1}^{n} (y_ i - b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ {ji}) x_ {ki}  = -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ {ki} = 0 \\\
\therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ {ki} = 0 @

@ SST = SSR + SSE + \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) \\\
\sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) &= \sum\limits_{i=1}^{n}2((y_ i - \hat{y}_ i) \hat{y}_ i - (y_ i - \hat{y}_ i) \bar{y}) \\\
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)\hat{y}_ i - 2 \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \bar{y} \\\
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ ji) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(\sum \limits_ {j=1}^{n} b_ j x_ ji) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(b_ 1 x_ {1i} + b_ 2 x_ {2i} \cdots b_ n x_ {ni}) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + b_ 1\sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)x_ {1i} + \cdots + b_ n\sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)x_ {ni} - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\
&= \; 0 \;\;\; \because \;  1) [^2] , 2) [^4] \\\
\therefore \;\; \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) = 0 @


@ \therefore \;\; SST = SSE + SSR @


[^1]: 해당 식을 $ b_ 0, b_ 1 $ 에 대한 함수 식이라 생각했을 때, 기울기가 0일 때 값이 최소/최대가 되기 때문. $ ( y_ i - b_ 0 - b_ 1 x_ i )^2 $ 은 $ b_ 0, b_ 1 $ 에 대한 2차방정식이라 생각할 수 있으며, 최고차항의 계수가 양수기 때문에 기울기가 0일때 최소값을 가짐       
[^2]: $ \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0  $    
[^3]: $ \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i = 0  $    
[^4]: $ \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ {ki} = 0  $    
