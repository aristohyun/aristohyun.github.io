---
layout: post
title: "DeepLearning 개념" 
description: "YOLO, You Only Look Once"
categories: [DeepLeaning]
tags: [DeepLeaning, YOLO, Object Detection]
use_math: true
redirect_from:
  - /2022/02/04
---

* Kramdown table of contents
{:toc .toc} 

# 기계학습, Machine Learning

> 주어진 데이터에서 사람이 특징을 추출해서 컴퓨터에게 학습(분류, 회귀) 등을 시키는 과정

![image](https://user-images.githubusercontent.com/32366711/153102750-3c7dc172-deb1-4bb4-849a-8dda61c6cd7c.png)

이때 특징 추출은 사람이 직접 해야함. 이 과정을 데이터 전처리 과정이라고 함

- 불필요한 데이터 삭제
- 중복 데이터 삭제
- 이상치 제거
- 표준화[^Stand], 정규화[^Reg] 등등

#### 예. 집 값 예측하기

<div class="multi-stage">
  <div class="stage" markdown=1>
  - 가격 (예측하고자 하는 값)
  - 집주소
  - 집 색깔
  - 지붕 모양
  - 우편번호
  - 현관 넓이
  - 뒷문 넓이
  - 차고 넓이
  - 차고에 넣을 수 있는 차 대수
  - 차고 건축 연도
  </div>
  <div class="stage" markdown=1>
  - 방 갯수
  - 욕실 갯수
  - 집 층수
  - 1층 넓이
  - 2층 넓이
  - 전체 넓이
  - 집 건축 연도
  - 집 리모델링 여부 
  - 집 리모델링 연도 
  - 벽난로 여부
  - 수영장 여부 ...
  </div>
</div>
  


![image](https://user-images.githubusercontent.com/32366711/153111346-7acf3919-733d-4d4a-9c10-e18a321e2e87.png)

##### 추출

![image](https://user-images.githubusercontent.com/32366711/153111350-327a5035-8f20-4c1c-8430-300e6a43e49c.png)


[^Stand]: 정규표현화
[^Reg]: 서로 다른 값의 범위를 0~1 사이로 동일하게 조정

# 딥러닝, Deep Learning

> 신경망과 은닉층의 도입으로 특징 추출조차 컴퓨터가 함

![image](https://user-images.githubusercontent.com/32366711/153101906-1bd26581-dd74-461a-8e1d-4238637fbda1.png)

이 히든레이어에서 뭘 어떤식으로 학습하는지 우리는 알 수 없음. 추측만 할 뿐

어떤 결론에 이르러 이런 값이 나왔는지 모름

이때 컴퓨터가 학습하는 대상은 w, 가중치

가중치를 늘리고 줄여서 필요한 데이터 특성을 추출함

가중치를 높여서 값을 살리고, 필요 없는건 가중치를 0으로 만들어 없앰

## 경사 하강법

> 오차에 대한 가중치 w의 비율을 계산하여, 조금씩 가중치를 조정하는 방식

@
w^{(t+1)} = w^{(t)} - \eta \frac{\partial E(w) }{\partial w}
@

값을 서서히 늘리거나 줄이면서 적정 값을 찾아야 하는데, 이때 사용하는 방법

즉 $\frac{\partial E(w^{(t)}) }{\partial w}$을 구해야 함

## Chain Rule, 연쇄 미분

@
y = f(g(x)) \\\ 
y = f(u), u = g(x)
@

@
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u}\frac{\partial u}{\partial x}
@

## 오차 역전파

[오차 역전파 설명](https://wikidocs.net/37406)

체인 룰에 따라 $\frac{\partial E(w^{(t)}) }{\partial w}$는 다음과 같이 쓸 수 있음

@
\frac{\partial E_ {total}}{\partial w_ k} = \frac{\partial E_ {total}}{\partial o_ i} * \frac{\partial o_ i}{\partial z_ i} * \frac{\partial z_ i}{\partial w_ {ij}}
@

- $o_ i = sigmoid(z_ i)$
- $z_ i = \sum \limits_ {j=0}^{N} w_ {ij} x_ {ij}$

이에 각각 미분식을 정리하면

@
\begin{align\*}
\frac{\partial E_ {total}}{\partial o_ i} &= \frac{\partial }{\partial o_ i} \frac{1}{2} (target_ i - o_ i)^2 \\\ 
&= -(target_ i - o_ i) \\\ 
\\\ 
\frac{\partial o_ i}{\partial z_ i} &= \frac{\partial sigmoid(z_ i)}{\partial z_ i} \\\ 
&= sigmoid(z_ i) * (1-sigmoid(z_ i)) \\\ 
&= o_ i * (1- o_ i) \\\ 
\\\ 
\frac{\partial z_ i}{\partial w_ {ij}} &= \frac{\partial }{\partial w_ {ij}} \sum \limits_ {j=0}^{N} w_ {ij} x_ {ij} \\\ 
&= x_ {ij}
\end{align\*}
@

따라서 $W$의 가중치로 계산을 했을 때, $E_ {total}$의 오차가 생겼다면, 변화율은 $ -(target - output) * (output * (1- output)) * (x_ j)$ 로 계산할 수 있다

이렇게 가중치를 변화해가며 계산을 하다가, 실제값에 가까워 질때까지, 혹은 아웃풋의 변화가 유의미하지 않을 때까지 진행한다

# CNN

> convolutional neural network, 합성곱 신경망          
> CNN 에서 조정하고자 하는 가중치는 필터이며, 필터링을 통해 해당 그림의 특징을 추출한다       

![image](https://user-images.githubusercontent.com/32366711/153126446-0ba3973f-5553-4b84-9f60-04c2e9a83ee5.png)

필터링이 끝날때 마다 활성화 함수를 거친 후 풀링을 진행한다

모든 필터링, 특징 추출이 끝났다면, 일반적인 신경망처럼 학습을 시킨다

다음 특징을 가졌을 때, 어떤 값이 나오는지를 학습시키는 것

이렇게 학습을 시킨 후, 결과값에 따라 오차역전파 방법을 통해 가중치(필터)를 조절하며 재학습 시킨다

~~~ python
model = Sequential()

# 특징 추출
model.add(Conv2D(12, kernel_size=(5, 5), activation='relu', input_shape=(120, 60, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 추출된 특징을 바탕으로 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

~~~
