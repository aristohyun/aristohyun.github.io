---
layout: post
title: "기계학습, 교차검증"
description: "교차검증"
categories: [Machine Learning]
tags: [Supervised Learning, Python, Regression, CV]
use_math: true
redirect_from:
  - /2021/06/26/
---

* Kramdown table of contents
{:toc .toc}

[집값 예측 : 회귀분석](https://www.kaggle.com/s1hyeon/house-price-regression/edit "캐글, House Price Predict"){: target="_blank"}    


# Cross Validation    

데이터셋은 두 종류로 나눌 수 있다    
학습을 시키기 위한 Training data와 Test data   

이 하나의 Training data로만 학습시켜서 Test data에 적합하게 모델링을 하다 보면    
딱 이 Test data에만 적합한 모델이 만들어 질 수 있다 (과최적화, Overfitting)    

이를 방지하기위해 교차검증을 한다    
그중 가장 일반적으로 사용되는 방법으로는 k-fold cross validation이 있다

## K-fold
교차검증시 학습용 데이터를 k개로 분할하여 이중 하나가 테스트 셋으로 활용
총 k번의 교차검증을 하는 셈
![k-fold](https://www.researchgate.net/profile/B_Aksasse/publication/326866871/figure/fig2/AS:669601385947145@1536656819574/K-fold-cross-validation-In-addition-we-outline-an-overview-of-the-different-metrics-used.jpg)

### 그외 CV 방법들    
- 홀드아웃 방법(Holdout method)    
- 리브-p-아웃 교차 검증(Leave-p-out cross validation)    
- 리브-원-아웃 교차 검증(Leave-one-out cross validation)    
- 계층별 k-겹 교차 검증(Stratified k-fold cross validation)    
