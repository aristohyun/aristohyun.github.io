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


# Neural Network

[Neural Network](https://ebbnflow.tistory.com/119)

[neural-networks](https://www.ibm.com/cloud/learn/neural-networks)

Neural networks reflect the behavior of the human brain, allowing computer programs to recognize patterns and solve common problems in the fields of AI, machine learning, and deep learning.


## ANN

딥러닝은 인공신경망(Artificial Neural Network)를 기초로 하고 있는데요. 인공신경망이라고 불리는 ANN은 사람의 신경망 원리와 구조를 모방하여 만든 기계학습 알고리즘 입니다.

인간의 뇌에서 뉴런들이 어떤 신호, 자극 등을 받고, 그 자극이 어떠한 임계값(threshold)을 넘어서면 결과 신호를 전달하는 과정에서 착안한 것입니다. 여기서 들어온 자극, 신호는 인공신경망에서 Input Data이며 임계값은 가중치(weight), 자극에 의해 어떤 행동을 하는 것은 Output데이터에 비교하면 됩니다.

신경망은 다수의 입력 데이터를 받는 입력층(Input), 데이터의 출력을 담당하는 출력층(Output), 입력층과 출력층 사이에 존재하는 레이어들(은닉층)이 존재합니다. 여기서 히든 레이어들의 갯수와 노드의 개수를 구성하는 것을 모델을 구성한다고 하는데, 이 모델을 잘 구성하여 원하는 Output값을 잘 예측하는 것이 우리가 해야할 일인 것입니다. 은닉층에서는 활성화함수를 사용하여 최적의 Weight와 Bias를 찾아내는 역할을 합니다. 


학습과정에서 파라미터의 최적값을 찾기 어렵다.
출력값을 결정하는 활성화함수의 사용은 기울기 값에 의해 weight가 결정되었는데 이런 gradient값이 뒤로 갈수록 점점 작아져 0에 수렴하는 오류를 낳기도 하고 부분적인 에러를 최저 에러로 인식하여 더이상 학습을 하지 않는 경우도 있습니다.

Overfitting에 따른 문제

학습시간이 너무 느리다.
은닉층이 많으면 학습하는데에 정확도가 올라가지만 그만큼 연산량이 기하 급수적으로 늘어나게 됩니다
느린 학습시간은 그래픽카드의 발전으로 많은 연산량도 감당할 수 있을 정도로 하드웨어의 성능이 좋아졌고, 오버피팅문제는 사전훈련을 통해 방지할 수 있게 되었습니다. 


## DNN

ANN기법의 여러문제가 해결되면서 모델 내 은닉층을 많이 늘려서 학습의 결과를 향상시키는 방법이 등장하였고 이를 DNN(Deep Neural Network)라고 합니다. DNN은 은닉층을 2개이상 지닌 학습 방법을 뜻합니다. 컴퓨터가 스스로 분류레이블을 만들어 내고 공간을 왜곡하고 데이터를 구분짓는 과정을 반복하여 최적의 구번선을 도출해냅니다. 많은 데이터와 반복학습, 사전학습과 오류역전파 기법을 통해 현재 널리 사용되고 있습니다.

그리고, DNN을 응용한 알고리즘이 바로 CNN, RNN인 것이고 이 외에도 LSTM, GRU 등이 있습니다.



## CNN

합성곱신경망 : Convolution Neural Network)

기존의 방식은 데이터에서 지식을 추출해 학습이 이루어졌지만, CNN은 데이터의 특징을 추출하여 특징들의 패턴을 파악하는 구조입니다. 이 CNN 알고리즘은 Convolution과정과 Pooling과정을 통해 진행됩니다. Convolution Layer와 Pooling Layer를 복합적으로 구성하여 알고리즘을 만듭니다.


## RNN

RNN(순환신경망 : Recurrent Neural Network)

RNN 알고리즘은 반복적이고 순차적인 데이터(Sequential data)학습에 특화된 인공신경망의 한 종류로써 내부의 순환구조가 들어있다는 특징을 가지고 있습니다. 순환구조를 이용하여 과거의 학습을 Weight를 통해 현재 학습에 반영합니다. 기존의 지속적이고 반복적이며 순차적인 데이터학습의 한계를 해결하연 알고리즘 입니다. 현재의 학습과 과거의 학습의 연결을 가능하게 하고 시간에 종속된다는 특징도 가지고 있습니다. 음성 웨이브폼을 파악하거나, 텍스트의 앞 뒤 성분을 파악할 때 주로 사용됩니다.


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
