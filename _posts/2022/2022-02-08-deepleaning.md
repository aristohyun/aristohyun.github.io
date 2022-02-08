---
layout: post
title: "YOLO, Object Detection Model" 
description: "YOLO, You Only Look Once"
categories: [DeepLeaning]
tags: [DeepLeaning, YOLO, Object Detection]
use_math: true
redirect_from:
  - /2022/02/04
---

* Kramdown table of contents
{:toc .toc} 

[오차 역전파 설명](https://wikidocs.net/37406)

# 경사 하강법

@
w^{(t+1)} = w^{(t)} - \eta \frac{\partial E(w^{(t)}) }{\partial w}
@

값을 서서히 늘리거나 줄이면서 적정 값을 찾아야 하는데, 이때 사용하는 방법

즉 $\frac{\partial E(w^{(t)}) }{\partial w}$을 구해야 함

# Chain Rule

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
&= -(target_ i - o_ i)
\end{align\*}
@

@
\begin{align\*}
\frac{\partial o_ i}{\partial z_ i} &= \frac{\partial sigmoid(z_ i)}{\partial z_ i} \\\ 
&= sigmoid(z_ i) * (1-sigmoid(z_ i)) \\\ 
&= o_ i * (1- o_ i)
\end{align\*}
@

@
\begin{align\*}
\frac{\partial z_ i}{\partial w_ {ij}} &= \frac{\partial }{\partial w_ {ij}} \sum \limits_ {j=0}^{N} w_ {ij} x_ {ij} \\\
&= x_ ij
\end{align\*}
@

따라서 $W$의 가중치로 계산을 했을 때, $E_ {total}$의 오차가 생겼다면, 변화율은 $ -(target - output) * (output * (1- output)) * (x_ j)$ 로 계산할 수 있다

이렇게 가중치를 변화해가며 계산을 하다가, 실제값에 가까워 질때까지, 혹은 아웃풋의 변화가 유의미하지 않을 때까지 진행한다

# CNN

cnn 에서 조정하고자 하는 가중치는 필터이며, 필터링은 해당 그림의 특징을 추출해 준다

필터링이 끝날때 마다 활성화 함수를 거친 후 풀링을 진행한다

모든 필터링, 특징 추출이 끝났다면, 일반적인 신경망처럼 학습을 시킨다

다음 특징을 가졌을 때, 어떤 값이 나오는지를 학습시키는 것이다

이렇게 학습을 시킨 후, 결과값에 따라 오차역전파 방법을 통해 가중치(필터)를 조절하며 재학습 시킨다

## YOLO
