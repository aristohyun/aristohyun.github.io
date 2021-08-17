---
layout: post
title: "기계학습, Gradient Decent"
description: "Gradient Decent, 경사하강법"
categories: [MachineLearning]
tags: [Machine Learning, Gradient Decent]
use_math: true
redirect_from:
  - /2021/08/01/
---

* Kramdown table of contents
{:toc .toc}      


[경사 하강법](https://angeloyeo.github.io/2020/08/16/gradient_descent.html){: target="_ blank"}


# 경사 하강법

> 함수의 최소값을 찾기 위해 사용 됨       

    
### 미분계수가 0인 지점을 찾으면 되지 않을까?                     

> 실제 분석에 사용되는 함수들은 닫힌 형태가 아니거나, 미분계수가 0이되는 지점이 하나 이상일 수 있으며          
> 미분계수보다는 경사하강법이 컴퓨터로 비교적 쉽게 구현할 수 있기 때문         
> 또한 데이터양이 매우 큰 경우 경사하강법과 같은 방법이 계산량 측면으로 효율적이다       

## 수식 유도

> 특정 위치 x에서 함수값이 커지는 중(기울기가 양수)이라면 음의 방향으로 x를 옮겨야 하며      
> 함수값이 작아지는 중(기울기가 음수)이라면 양의 방향으로 x를 옮기면 된다

@
x_ {i+1} = x_ i - 이동거리 \times  기울기
@

이때 이동거리는 Gradient의 크기를 이용하게 된다              
이동거리에 사용할 값을 gradient의 크기와 비례하는 factor를 이용하면            
현재 x의 값이 극소값에서 멀 때는 많이 이동하고, 극소값에 가까워졌을 때는 조금씩 이동할 수 있게 된다.           


@
x_ {i+1} = x_i - \alpha \frac{d f}{d x_}(x_ i)
@

![image](https://cdn.hackernoon.com/hn-images/1*ZmzSnV6xluGa42wtU7KYVA.gif)


#### 적절한 크기의 step size 필요

![image](https://user-images.githubusercontent.com/32366711/128164649-d136b81f-f408-4fa6-b1b2-567668fa79c7.png)
