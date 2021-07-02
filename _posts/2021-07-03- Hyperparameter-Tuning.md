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

# Hyperparameter    
- k at KNN    
- linear regression parameters(coefficients)    
- alpha(lambda) at Ridge and Lasso    
- Random forest parameters like max_depth     

## Hyperparameter tuning      
- try all of combinations of different parameters       
- fit all of them      
- measure prediction performance     
- see how well each performs    
- finally choose best hyperparameters    

### GridSearchCV    
grid: K is from 1 to 50(exclude)    
GridSearchCV takes knn and grid and makes grid search. It means combination of all hyperparameters. Here it is k.    

~~~ python    
# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x,y)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
~~~    
