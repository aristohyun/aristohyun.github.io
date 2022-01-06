---
layout: post
title: "기계학습, Hyperparameter Tuning"
description: "Hyperparameter Tuning"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Hyperparameter Tuning]
use_math: true
redirect_from:
  - /2021/07/03/
---

* Kramdown table of contents
{:toc .toc}

# Parameter[^1]   
> 모델 내부에서 결정되는 변수이며, 그 값은 데이터로부터 결정됨     

- 데이터를 통해 구해지며      
- 모델 내부적으로 결정되는 값이며, 사용자에 의해 조정되지 않음      
- 사용자가 직접 설정하는 것이 아니라 모델링에 의해 자동으로 결정되는 값      
  

# Hyperparameter    
> 모델 외부에 있으며, 데이터를 통해서는 값을 예측할 수 없음   
> 모델 파라미터를 추정하는 데 도움이 되는 프로세스에서 자주 사용됨    

- 사용자가 직접 설정하며
- 정해진 최적의 값이 없기에 휴리스틱[^2]등을 통해 값을 구하는 등 
- 주어진 모델링에 맞춰 조절해야 함

## 예    
- k at KNN    
- linear regression parameters(coefficients)    
- alpha(lambda) at Ridge and Lasso    
- Random forest parameters like max_depth     

## Hyperparameter tuning      
> 파라미터의 모든 조합들을 시도해서 예측 성능을 측정해 보며, 
> 어떤게 가장 좋은 성능을 내는지 확인하는 것     

### GridSearchCV    
> Hyperparameter 후보군들을 넣어서 어떤게 가장 성능이 좋은지 계산하는 클래스    

<br />
- 앞에서는 K의 최적의 값을 찾기 위해서 반복문을 통해 값을 하나씩 확인했음    

~~~ python
# k from 1 to 25(exclude)
for i, k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train, y_train)))   
~~~    

- 위의 방법 대신 GridSearchV로 찾는 것     

~~~ python    
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(X_train,Y_train)# Fit    

# Print hyperparameter    
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
~~~    

![image](https://user-images.githubusercontent.com/32366711/124721688-4991e380-df44-11eb-9ad1-1ec158fe05cf.png)    

<br />
- 로지스틱회귀에서 가중치(C)와 패널티를 l1으로 할것인지 l2로 할것인지 또한 설정 가능    

~~~ python    

# grid search cross validation with 2 hyperparameter
# 1. hyperparameter is C:logistic regression regularization parameter
# 2. penalty l1 or l2
# Hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train ,test_size = 0.3,random_state = 12)
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=3)
logreg_cv.fit(x_train,y_train)

# Print the optimal parameters and best score
print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))
~~~    

![image](https://user-images.githubusercontent.com/32366711/124724183-bdcd8680-df46-11eb-87a4-68f038aaa6e7.png)


[^1]: [Machine Learning Mastery, What is the Difference Between a Parameter and a Hyperarameter?](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/){: target="_ blank"}

[^2]: 휴리스틱(heuristics) 또는 발견법(發見法)이란 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 사람들이 빠르게 사용할 수 있게 보다 용이하게 구성된 간편추론의 방법이다
