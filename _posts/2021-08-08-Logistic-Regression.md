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
&= b + px_1w_1 + px_2w_2 + ... + px_{4096} * w_{4096} \\\ 
\hat y &= sigmoid(z) 
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

> From weights and bias to cost
> 뉴럴 네트워크 모델의 입력층부터 출력층까지 순서대로 변수들을 계산하고 저장하는 것을 의미함

Forward propagation steps:     
- find z = w.T * x + b         
- $\hat y$  = sigmoid(z)         
- loss(error) = loss(y, $\hat y$)          
- cost = sum(loss)        

~~~ python
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_hat = sigmoid(z) # probabilistic 0-1
    loss = -y_train*np.log(y_hat)-(1-y_train)*np.log(1-y_hat)
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
    y_hat = 1/(1+np.exp(-z))
    return y_hat
~~~


### Loss(error) Function [^loss]

> Is our prediction is correct and how do we check whether it is correct or not?        
> The answer is with loss(error) function:     

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

> 코스트 함수는 각각의 데이터에서 생기는 총 손실의 합이 된다       

@
Cost = \frac{1}{m} \sum \limits_ {i=1}^{m} Loss_ i
@


## Optimization Algorithm with Gradient Descent

### Gradient Desecnt

> 코스트가 높다는 것은 잘못된 예측을 의미하기에 코스트를 최대한 줄여야 한다            
> 그러기위해 weights 와 bias를 업데이트하며 적절한 값을 찾는다           

@
w := w - \alpha \frac {\partial Cost(w,b)}{\partial (w,b)} 
@

### Backward Propagation

> From cost to weights and bias to update them

@
\frac {\partial Cost}{\partial w} = \frac{1}{m} x(\hat y - y)^T \\\ 
\frac {\partial Cost}{\partial b} = \frac{1}{m} \sum \limits_ {i=1}^{m} (\hat y - y)
@

~~~ python
# In backward propagation we will use y_hat that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_hat = sigmoid(z)
    loss = -y_train*np.log(y_hat)-(1-y_train)*np.log(1-y_hat)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_hat-y_train).T)))/x_train.shape[1]  # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_hat-y_train)/x_train.shape[1]                    # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost, gradients
~~~

#### Differential : Chain Rule

@
\begin{align\*}
z &= w^T x + b \\\ 
\hat y &= 1/(1+exp(-z)) \\\ 
Loss &= -(1 - y) log (1 - \hat y) - y log \hat y \\\ 
Cost &= \frac{1}{m} \sum \limits_ {i=1}^{m} (-(1 - y) log (1 - \hat y) - y log \hat y )
\end{align\*}
@

<br/>

@
\frac {\partial Cost}{\partial w} = \frac{1}{m} \sum \limits_ {i=1}^{m} \frac {\partial Loss}{\partial w} = \frac{1}{m} \sum \limits_ {i=1}^{m} \frac {\partial Loss}{\partial \hat y} \frac {\partial \hat y}{\partial z} \frac {\partial z}{\partial w} \\\ 
\frac {\partial Cost}{\partial b} = \frac{1}{m} \sum \limits_ {i=1}^{m} \frac {\partial Loss}{\partial b} = \frac{1}{m} \sum \limits_ {i=1}^{m} \frac {\partial Loss}{\partial \hat y} \frac {\partial \hat y}{\partial z} \frac {\partial z}{\partial b}
@

<br/>

@
\begin{align\*}
\frac {\partial Loss}{\partial \hat y} &= \frac {\partial }{\partial \hat y} (-(1 - y) log (1 - \hat y) - y log \hat y) \\\ 
&= -(y\frac {\partial }{\partial \hat y} log \hat y + (1-y)\frac {\partial }{\partial \hat y}log(1-\hat y) ) \\\ 
\frac {\partial Loss}{\partial \hat y} &= -(y \frac{1}{\hat y} - (1-y)\frac{1}{1-\hat y})
\end{align\*}
@

<br/>

@
\begin{align\*}
\frac {\partial \hat y}{\partial z} &= \frac {\partial }{\partial z} (\frac{1}{1+e^{-z}}) = \frac {\partial }{\partial z}(1+e^{-z})^{-1} \\\ 
&= -(1+e^{-z})^{-2}\frac {\partial }{\partial z}(e^{-z}) = -(1+e^{-z})^{-2}(-e^{-z}) \\\ 
&= \frac{e^{-z}}{(1+e^{-z})^2} \\\ 
&= \frac{1}{(1+e^{-z})} \frac{e^{-z}}{(1+e^{-z})} \\\ 
&= \frac{1}{1+e^{-z}} (1- \frac{1}{1+e^{-z}}) \\\ 
\frac {\partial \hat y}{\partial z} &= \hat y(1-\hat y)
\end{align\*}
@

<br/>

@
z = w^Tx+b \\\ 
\begin{align\*}
\frac {\partial z}{\partial w} &= x \\\ 
\frac {\partial z}{\partial b} &= 1 
\end{align\*}
@

<br/>

@
\begin{align\*}
\frac {\partial Loss}{\partial w} &= \frac {\partial Loss}{\partial \hat y} \frac {\partial \hat y}{\partial z} \frac {\partial z}{\partial w} \\\ 
&= -(y \frac{1}{\hat y} - (1-y)\frac{1}{1-\hat y}) \; \hat y(1-\hat y) \; x \\\ 
&= (\hat y - y)x \\\ 
\frac {\partial Cost}{\partial w} &= \frac{1}{m} \sum \limits_ {i=1}^{m} (\hat y_ i - y_ i)x_ i \\\ 
&= \frac{1}{m} x(\hat y - y)^T
\end{align\*}
@

<br/>

@
\begin{align\*}
\frac {\partial Loss}{\partial b} &= \frac {\partial Loss}{\partial \hat y} \frac {\partial \hat y}{\partial z} \frac {\partial z}{\partial b} \\\ 
&= -(y \frac{1}{\hat y} - (1-y)\frac{1}{1-\hat y}) \; \hat y(1-\hat y) \; 1 \\\ 
&= \hat y - y \\\ 
\frac {\partial Cost}{\partial b} &= \frac{1}{m} \sum \limits_ {i=1}^{m} ( \hat y_ i - y_ i )
\end{align\*}
@

### Updating parameters     

~~~ python
# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
    
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
    
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
~~~

### Predict

~~~ python

def predict(w, b, x_test):

    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
    
# predict(parameters["weight"],parameters["bias"],x_test)

~~~

~~~ python

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_weights_and_bias(dimension)
    
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)

~~~

![image](https://user-images.githubusercontent.com/32366711/129881158-9fcea686-a1fe-439c-ab52-62bc7bc826fd.png)


## Logistic Regression with Sklearn

[sklearn : Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

~~~ python
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
~~~



[^sigmoid]: 시그모이드 함수, S자형 곡선 또는 시그모이드 곡선을 갖는 수학 함수. ![image](https://user-images.githubusercontent.com/32366711/128862627-1c0408c2-19b3-4cd7-9aa7-61c8901a0e4e.png){: width="300"}
 
[^loss]: ![image](https://user-images.githubusercontent.com/32366711/128870484-9346e71e-3f94-4501-9d52-2cc5b9a82c56.png)
