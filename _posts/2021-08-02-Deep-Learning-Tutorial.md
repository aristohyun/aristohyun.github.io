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


# Introduction

> Deep learning: 데이터를 통해 기능을 직접 학습하는 기계 학습 기법 중 하나        
> Why deep learning: 데이터 양이 증가하면, 머신러닝 기술은 성능 면에서 부족하지만, 딥러닝은 더 나은 성능(정확도 등)을 제공합니다.

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



[^ps]: 실제로 205개의 샘플은 딥러닝에 매우 매우 적은 양입니다. 하지만 이것은 튜토리얼이기 때문에 크게 문제가 되지 않습니다.
