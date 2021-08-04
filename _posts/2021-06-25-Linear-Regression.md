---
layout: post
title: "기계학습, Linear Regression"
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

> 이론이나 경험적 근거에 의해 설정된 변수들간의 함수관계가 유의한지 알아보는 통계분석 방법       
> 회귀분석이란, 변수들 사이의 관계를 추정하는 분석방법이며, 독립변수가 주어졌을 때 종속변수를 예측하고자 한다    
> 주어진 데이터들을 가장 잘 설명하는 관계식(회귀식), 하나의 선(회귀선)을 구하는 방법이라고 할 수 있다


## 역사    

> Galton이 부모와 자식의 키에 대한 조사를 하며 이를 그래프로 그려보았더니, 부모의 키와 유사한 경향을 보였으며, 자식들의 키는 평균으로 회귀하고 있음을 찾아내었음            
> 이후 갈톤과 다른 통계학자들이 변수간 상관관계를 계량화하고 데이터를 적합하는 선을 결정하는 방법론들을 확립하였으며,             
> 그제서야 regression이라는 용어가 비로소 우리가 현재 회귀분석이라고 부르는 통계 분석 방법과 관련있는 것이 되었다고 함             

> 이 평균으로의 회귀 현상은 모든 자연, 사회적 현상에서 보이기 때문에 정규분포를 이룬다고 가정하여  검정할 수 있음         

![image](https://user-images.githubusercontent.com/32366711/128171679-7122aedf-1b70-4af5-9848-1c5234d27e8a.png){: width="500"}{: .aligncenter}


## 회귀 분석의 4단계

1. 이론의 가정    
2. 회귀직선 그리기 : 최소제곱법, $R^2$    
3. 가설검정 : 모수 추정 (실제 선형관계 추정)    
4. 평균값에 대한 예측, 개별값에 대한 예측    


## 이론의 가정   

### 회귀분석의 기본 가정

1. 두 변수 간 선형관계가 있어야한다
2. 표본추출이 무작위하게 이루어져야 한다
3. X의 값이 두개 이상이여야 한다
4. Zero-conditional Mean : 오차들의 평균은 0이되어야 한다
5. 등분산성, Homoskedaticity : 오차들이 같은 정도로 퍼져있어야 한다
6. 독립성, Independence : 오차항들끼리 독립적이어야 한다
7. 정규성, Normalty : 오차들끼리는 정규분포를 이루어야 한다

### 선형회귀의 기본 가정

1. Linearity (선형성)
2. Independence (독립성)
3. Normality (정규성)
4. Homoskedaticity (등분산성)


### 왜 회귀분석은 정규분포를 가정하는가

[선형 회귀 모형 가정을 왜 하는가](https://laoonlee.tistory.com/5) 
[가설검정](https://kkokkilkon.tistory.com/36)          

회귀 분석은 회귀 모형을 추정한 이후 회귀 모형이 잘 맞는지 모형 검정과 계수 검정을 필요로 합니다.            
모형 검정과 계수 검정 등의 가설검정을 하기 위해서는 분포 가정이 필요 하여, 이때 사용하기 위함            
F-검정           

[회귀분석의 표준 가정](https://hooni-playground.com/1225/)             
[가우스 마르코프 정리](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gdpresent&logNo=221138157186)                

가우스-마르코프 정리[^GMT]에서는           
잔차의 기대값이 0이고 서로 다른 두 잔차의 공분산이 0이며(두 잔차가 독립적이다면),      
잔차가 등분산성을 만족한다면          
OLS[^OrdinaryLeastSquare]가 BLUE[^BestLinearUnbiasedEstimator] 하며           
이 회귀분석은 믿을만 하다는 것을 알 수 있다          

즉 잔차가 정규성을 만족하지 않더라도 위의 조건만 만족한다면 OLS가 BLUE하다고 할 수 있다


## 회귀 직선 그리기

![단순 선형회귀](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile7.uf.tistory.com%2Fimage%2F997E924F5CDBC1A6283C93)    
기본적으로 단순 선형 회귀식은 다음과 같다    
$Y = b_{0} + b_{1}X$       

독립변인이 늘어나면 다음과 같아진다 (다중 선형 회귀식)    
$Y = b_{0} + b_{1}X_{1} + b_{2}X_{2} + b_{3}X_{3} ...$    

## 결정계수, R-Squared, $R^2$    

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

### SST = SSE + SSR 유도      

[참고](https://datalabbit.tistory.com/51){:target="_ blank"} <br/>   

회귀분석에서는 SSR ( Residual Sum of Square )이 최소가 되도록 해야함    

$ \hat{y} = b_0 + b_1 x $

$ SSR = \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ i)^2 = \sum\limits_ {i=1}^{n}(y_ i - b_0 - b_1 x_ i)^2 $

<br/>

$ \Rightarrow \;\;\;\; b_ 0, b_ 1 $에 대하여 미분했을 때 0이 되야 SSR이 최소값 [^1]    

$ 1) \;\; \frac{\partial }{\partial b_ 0} \sum\limits_ {i=1}^{n}(y_ i - b_0 - b_1 x_ i)^2 \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - b_0 - b_1 x_ i) \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 \\\ 
\therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 $ 
  
<br/>
  
$ 2) \;\; \frac{\partial }{\partial b_ 1} \sum\limits_ {i=1}^{n}(y_ i - b_0 - b_1 x_ i)^2 \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - b_0 - b_1 x_ i) x_ i  \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i = 0 \\\ 
\therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i = 0 $     

<br/>

$ \begin{align\*}SST &= \sum\limits_{i=1}^{n}(y_ i - \bar{y})^{2} \\\ 
&= \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ {i} + \hat{y}_ i - \bar{y})^2 \\\ 
&= \sum\limits_ {i=1}^{n}((y_ i - \hat{y}_ {i})^2 + 2(y_ i - \hat{y}_ {i})(\hat{y}_ {i} - \bar{y}) +(\hat{y}_ {i} - \bar{y})^2) \\\ 
&= \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ {i})^2  +  \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ {i} - \bar{y})  +  \sum\limits_ {i=1}^{n}(\hat{y}_ {i} - \bar{y})^2 \\\ 
&= SSR + SSE + \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y})\end{align\*}$

$ \begin{align\*}\sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) &= \sum\limits_{i=1}^{n}2((y_ i - \hat{y}_ i) \hat{y}_ i - (y_ i - \hat{y}_ i) \bar{y}) \\\ 
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)\hat{y}_ i - 2 \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \bar{y} \\\ 
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(b_0 + b_1 x_ i) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + 2 b_1 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= \; 0 \end{align\*}$ [^2][^3]    

$\therefore \;\; \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) = 0 $

$\therefore \;\; SST = SSE + SSR $

<br/>

### SST = SSR + SSE 일반화     

$ \begin{align\*}\hat{y} &= b_ 0 + b_ 1 x_ 1 + b_ 2 x_ 2 \cdots b_ n x_ n \\\ 
&= b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ j \end{align\*}$

$ SSR = \sum\limits_ {i=1}^{n}(y_ i - \hat{y}_ i)^2 = \sum\limits_ {i=1}^{n}(y_ i - b_ 0 - \sum \limits_ {j=1}^{n} b_ j x_ {ji} )^2 $

<br/>

$ \Rightarrow \;\;\;\; b_ i $에 대하여 각각 미분했을 때 0이 되야 SSR이 최소값을 가짐 [^1]     

$ 1) \; \frac{\partial }{\partial b_ 0} \sum\limits_ {i=1}^{n}(y_ i - b_ 0 - \sum \limits_ {j=1}^{n} b_ j x_ {ji})^2 \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - b_ 0 - \sum \limits_ {j=1}^{n} b_ j x_ {ji})  \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 \\\ 
\therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0 $

<br/>

$ 2) \; \frac{\partial }{\partial b_ k} \; \sum\limits_ {i=1}^{n}(y_ i - b_ 0 - \sum \limits_ {j=1}^{n} b_ j x_ {ji})^2  \;\;\;\;\;\; (1\leq k\leq n \;,\;\; k\in \mathbb{\mathbb{N}}) \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - b_ 0 - \sum \limits_ {j=1}^{n} b_ j x_ {ji}) x_ {ki} \\\ 
\;\; = -2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ {ki} = 0 \\\ 
\therefore \;\; \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ {ki} = 0 $

<br/>

$ SST = SSR + SSE + \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) $

$ \begin{align\*}\sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) &= \sum\limits_{i=1}^{n}2((y_ i - \hat{y}_ i) \hat{y}_ i - (y_ i - \hat{y}_ i) \bar{y}) \\\ 
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)\hat{y}_ i - 2 \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \bar{y} \\\ 
&= 2 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(b_ 0 + \sum \limits_ {j=1}^{n} b_ j x_ ji) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(\sum \limits_ {j=1}^{n} b_ j x_ {ji}) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)(b_ 1 x_ {1i} + b_ 2 x_ {2i} \cdots b_ n x_ {ni}) - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= 2 b_0 \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) + b_ 1\sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)x_ {1i} + \cdots + b_ n\sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i)x_ {ni} - 2\bar{y} \sum\limits_{i=1}^{n} (y_ i - \hat{y}_ i) \\\ 
&= \; 0 \end{align\*} $ [^2][^4]    

$ \therefore \;\; \sum\limits_ {i=1}^{n}2(y_ i - \hat{y}_ i)(\hat{y}_ i - \bar{y}) = 0 $

$ \therefore \;\; SST = SSE + SSR $

### 일반화 다른 방법

[막힘](http://ko.stucarrot.wikidok.net/wp-c/%EA%B2%BD%EC%A0%9C%EA%B3%B5%EB%B6%80:%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C/View?ci=2#wk_cTitle1164)

![image](https://user-images.githubusercontent.com/32366711/125510770-466e728d-01f9-4488-b168-1b054ef19353.png)


![image](https://user-images.githubusercontent.com/32366711/125501083-31dd1d9c-2891-499b-b14a-678a58713fdb.png)

![image](https://user-images.githubusercontent.com/32366711/125501089-6c79468b-ff0b-4d87-bf9d-f73ec076e256.png)


[계량경제학](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=dhkdwnddml&logNo=220156712894)              
[간토끼1](https://datalabbit.tistory.com/50)             
[간토끼2](https://datalabbit.tistory.com/51)               


[^1]: 해당 식을 $ b_ 0, b_ 1 $ 에 대한 함수 식이라 생각했을 때, 기울기가 0일 때 값이 최소/최대가 되기 때문. $ ( y_ i - b_ 0 - b_ 1 x_ i )^2 $ 은 $ b_ 0, b_ 1 $ 에 대한 2차방정식이라 생각할 수 있으며, 최고차항의 계수가 양수기 때문에 기울기가 0일때 최소값을 가짐        
[^2]: $ \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) = 0  $        
[^3]: $ \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ i = 0  $          
[^4]: $ \sum\limits_ {i=1}^{n} (y_ i - \hat{y}_ i) x_ {ki} = 0  $       
[^GMT]: 가우스-마르코프 정리란,  회귀분석의 가정이 성립할 때 최소제곱추정량이 가장 작은 분산을 갖는 효율적인 추정량임을 말할 수 있다           
[^OrdinaryLeastSquare]: 최소 자승법, 오차 제곱의 합이 최소가 되게 하는 방법        
[^BestLinearUnbiasedEstimator]: 최량 선형 불편 추정량,  Linear(선형)이며 Unbiased(편향되지 않은)인 Estimator(추정치) 중에 가장 좋은(best, 분산의 값이 가장 작은) 방법            
