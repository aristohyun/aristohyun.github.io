---
layout: post
title: "딥러닝, Logistic Regression"
description: "Logistic Regression, 로지스틱 회귀"
categories: [DeepLearning]
tags: [Deep Learning, Logistic Regression]
use_math: true
redirect_from:
  - /2021/08/08/
---

* Kramdown table of contents
{:toc .toc}      


# Logistic Regression

- logistic regression is actually a very simple neural network.

~~~ python
# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
~~~

![image](https://user-images.githubusercontent.com/32366711/128638215-a79c9acb-8e2b-4f33-9960-fccbca8c5bd8.png)

~~~ python
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)
~~~

![image](https://user-images.githubusercontent.com/32366711/128638233-b3ec7902-2fab-4ba4-a377-c974bac650b8.png)


![image](https://user-images.githubusercontent.com/32366711/128638111-b25de13c-7149-442d-bc14-cab76fdb90bb.png)


## Computation Graph

왜 계산 그래프를 쓸까?

1. 데이터와 데이터 하나에 관여하는 요소들(차원)이 너무 많은데, 이들의 수식 관계를 가장 잘 보여줄 수 있기 때문               
2. 순전파와 역전파에 대한 이해와 계산이 용이해짐

> Even relatively “simple” deep neural networks have hundreds of thousands of nodes and edges; it’s quite common for a neural network to have more than one million edges. Try to imagine the function expression for such a computational graph… can you do it? How much paper would you need to write it all down? This issue of scale is one of the reasons computational graphs are used. [Deep Neural Networks As Computational Graphs](https://medium.com/tebs-lab/deep-neural-networks-as-computational-graphs-867fcaa56c9)

[Computational Graphs: Why](https://medium.com/neuromation-blog/neuronuggets-what-do-deep-learning-frameworks-do-exactly-6615cd55d618)

![image](https://user-images.githubusercontent.com/32366711/128638011-ccee8e52-f7f7-4f4d-83cb-54008f0ad01d.png)

- Weights: coefficients of each pixels    
- Bias: intercept    

@
\begin{align\*}
Z &= W^T X + b \\\ 
&= z = b + px_1w_1 + px_2w_2 + ... + px_{4096} * w_{4096} \\\ 
y_{head} &= sigmoid(z) 
\end{align\*}
@


## Initializing parameters     

~~~ python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
~~~

4096개의 차원에 초기 weight는 0.01, bias는 0으로 설정


## Forward Propagation

뉴럴 네트워크 모델의 입력층부터 출력층까지 순서대로 변수들을 계산하고 저장하는 것을 의미함

> Forward propagation steps:     
> find z = w.T * x + b         
> y_head = sigmoid(z)         
> loss(error) = loss(y,y_head)          
> cost = sum(loss)        

~~~ python
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z) # probabilistic 0-1
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling, 전체 차원의 갯수로 나눠줌. 로스의 평균
    return cost 
~~~


### Sigmoid Function[^sigmoid]

> Z값을 구시그모이드 함수에 넣으면, 이게 y인지에 대한 확률이 나옴 

@
S(x) = \frac {1}{1+e^{-x}} = \frac {e^x}{e^x + 1}
@

~~~ python
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
~~~

### Loss(error) Function [^loss]

> 이렇게 y 추정값을 구했을 때, 이게 얼만큼 맞는지 Then e.g y_head became 0.9 that is bigger than 0.5 so our prediction is image is sign one image. Okey every thing looks like fine. But, is our prediction is correct and how do we check whether it is correct or not? The answer is with loss(error) function:

$y$ : 실제값         
$\hat y$ : 추정값(확률)         

@
 -(1 - y) log (1 - \hat y) - y log \hat y
@

y가 1일 때,
y햇을 1이라고 예측하면 에러 값은 0. 에러가 없음
y가 1일때 y햇을 0이라고 예측하면 에러값은 - log 0 = Infinite, 무한대로 커짐. 즉 에러가 굉장히 큼
에러를 작게해야함 => 이 모든 에러의 값을 합친 코스트 값을 최소로 해야함

### Cost Function

After that, the cost function is summation of loss function. Each image creates loss function. Cost function is summation of loss functions that is created by each input image.
Lets implement forward propagation.

----------

## Optimization Algorithm with Gradient Descent

### Backward Propagation

### Updating parameters

## Logistic Regression with Sklearn

## Summary and Questions in Minds

## [순전파, 역전파](https://ko.d2l.ai/chapter_deep-learning-basics/backprop.html)


[^sigmoid]: 시그모이드 함수, S자형 곡선 또는 시그모이드 곡선을 갖는 수학 함수. ![image](https://user-images.githubusercontent.com/32366711/128862627-1c0408c2-19b3-4cd7-9aa7-61c8901a0e4e.png){: width="300"}
 
[^loss]: ![image](https://user-images.githubusercontent.com/32366711/128870484-9346e71e-3f94-4501-9d52-2cc5b9a82c56.png)
