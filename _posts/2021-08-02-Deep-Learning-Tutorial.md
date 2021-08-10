---
layout: post
title: "딥러닝, 튜토리얼"
description: "딥러닝"
categories: [DeepLearning]
tags: [Deep Learning]
use_math: true
redirect_from:
  - /2021/08/02/
---

* Kramdown table of contents
{:toc .toc}      


# DeapLearning[^deep]
  
> 딥 러닝 모델은 인간이 결론을 내리는 방식과 유사한 논리 구조를 사용하여 데이터를 지속적으로 분석하도록 설계되었습니다.               
> 이를 달성하기 위해 딥 러닝 애플리케이션은 '인공 신경망'이라는 계층화된 알고리즘 구조를 사용합니다.                         

> 기본 머신 러닝 모델은 그 기능이 무엇이든 점진적으로 향상되는데, 여전히 약간의 안내가 필요합니다. AI 알고리즘이 부정확한 예측을 반환하면 엔지니어가 개입하여 조정해야 합니다.           
> 딥 러닝 모델을 사용하면 알고리즘이 자체 신경망을 통해 예측의 정확성 여부를 스스로 판단할 수 있습니다.       

![image](https://user-images.githubusercontent.com/32366711/128165264-fba6e635-62e2-431a-b8b0-2b3764cf8a3e.png)

- What is amount of big : 직관적으로 100만 개의 샘플이면 "대용량 데이터"라고 말할 수 있습니다            
- Usage fields of deep learning: 음성 인식, 이미지 분류, 자연어 행렬(nlp) 또는 추천 시스템             
- What is difference of deep learning from machine learning:        
  + 머신러닝은 딥러닝을 다룹니다       
  + 머신러닝은 수동으로 기계 학습시켜야 하지만              
  + 딥러닝은 데이터를 통해 직접 기능을 학습합니다              

![image](https://user-images.githubusercontent.com/32366711/128166064-a7d49b3f-6219-4d26-a830-cdc7cd27119d.png)


# Overview the Data Set

> 수화 숫자 데이터 세트      

이 데이터에는 2062개의 수화 숫자 이미지가 있으며                   
튜토리얼을 시작할 때 단순성을 위해 기호 0과 1만 사용하게 됨                   
데이터에서 부호 0은 인덱스 204와 408 사이에 있으며,(205개)                   
또한 부호 1은 인덱스 822와 1027 사이입니다.(206개)                   
따라서 각 클래스(라벨)에서 205개의 샘플을 사용할 것입니다[^ps]                   
X는 이미지 배열(0 및 기호 1개)이고 Y는 레이블 배열(0 및 1)로 배열을 만들어 사용하게 됩니다                   



[^deep]: 층 기반 표현 학습 layered representations learning 또는 계층적 표현 학습 hierarchical representations learning. 연속된 층으로 표현을 학습한다는 개념을 나타냅니다. 데이터로부터 모델을 만드는 데 얼마나 많은 층을 사용했는지가 그 모델의 깊이가 됩니다
[^ps]: 실제로 205개의 샘플은 딥러닝에 매우 매우 적은 양입니다. 하지만 이것은 튜토리얼이기 때문에 크게 문제가 되지 않습니다.
