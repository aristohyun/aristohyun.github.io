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

> Even relatively “simple” deep neural networks have hundreds of thousands of nodes and edges; it’s quite common for a neural network to have more than one million edges. Try to imagine the function expression for such a computational graph… can you do it? How much paper would you need to write it all down? This issue of scale is one of the reasons computational graphs are used. [Deep Neural Networks As Computational Graphs](https://medium.com/tebs-lab/deep-neural-networks-as-computational-graphs-867fcaa56c9)

![image](https://user-images.githubusercontent.com/32366711/128638011-ccee8e52-f7f7-4f4d-83cb-54008f0ad01d.png)

- Weights: coefficients of each pixels    
- Bias: intercept    

@
\begin{align\* }
Z $= W^T X + b \\\ 
$=  z = b + px1w1 + px2w2 + ... + px4096 * w4096 \\\ 
y_head &= sigmoid(z)
\end{align\* }
@

Sigmoid function makes z probability. 


## Initializing parameters

The first step is multiplying each pixels with their own weights.            
The question is that what is the initial value of weights?                

There are some techniques that I will explain at artificial neural network but for this time initial weights are 0.01.           
Okey, weights are 0.01 but what is the weight array shape? As you understand from computation graph of logistic regression, it is (4096,1)           
Also initial bias is 0.            
Lets write some code. In order to use at coming topics like artificial neural network (ANN), I make definition(method).       

~~~ python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
~~~

4096개의 차원에 초기 weight는 0.01, bias는 0으로 설정


## Forward Propagation

z = (w.T)x + b => 
in this equation we know x that is pixel array, we know w (weights) and b (bias) so the rest is calculation. (T is transpose)
Then we put z into sigmoid function that returns y_head(probability). When your mind is confused go and look at computation graph. Also equation of sigmoid function is in computation graph.
Then we calculate loss(error) function.
Cost function is summation of all loss(error).
Lets start with z and the write sigmoid definition(method) that takes z as input parameter and returns y_head(probability)


> Forward propagation steps:     
> find z = w.T*x+b         
> y_head = sigmoid(z)         
> loss(error) = loss(y,y_head)          
> cost = sum(loss)        

~~~ python
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z) # probabilistic 0-1
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    return cost 
~~~


### Sigmoid Function[^sigmoid]

@
S(x) = \frac {1}{1+e^{-x}} = \frac {e^x}{e^x + 1}
@

~~~ python
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
~~~

### Loss(error) Function

Lets make example, I put one image as input then multiply it with their weights and add bias term so I find z. Then put z into sigmoid method so I find y_head. Up to this point we know what we did. Then e.g y_head became 0.9 that is bigger than 0.5 so our prediction is image is sign one image. Okey every thing looks like fine. But, is our prediction is correct and how do we check whether it is correct or not? The answer is with loss(error) function:

y = 실제값
y hat = 추정값(확률)
@
- (1 - y) log (1 - \hat y) - y log \hat y
@

y가 1일 때,
y햇을 1이라고 예측하면 에러 값은 0. 에러가 없음
y가 1일때 y햇을 0이라고 예측하면 에러값은 - log 0 = Infinite, 무한대로 커짐. 즉 에러가 굉장히 큼
이 모든 에러의 값을 합친 코스트 값을 최소로 해야함


It says that if you make wrong prediction, loss(error) becomes big. DENKLEM DUZELTME
Example: our real image is sign one and its label is 1 (y = 1), then we make prediction y_head = 1. When we put y and y_head into loss(error) equation the result is 0. We make correct prediction therefore our loss is 0. However, if we make wrong prediction like y_head = 0, loss(error) is infinity.

### Cost Function

After that, the cost function is summation of loss function. Each image creates loss function. Cost function is summation of loss functions that is created by each input image.
Lets implement forward propagation.

----------

## Optimization Algorithm with Gradient Descent

### Backward Propagation

### Updating parameters

## Logistic Regression with Sklearn

## Summary and Questions in Minds


[^sigmoid]: 시그모이드 함수, S자형 곡선 또는 시그모이드 곡선을 갖는 수학 함수. 
