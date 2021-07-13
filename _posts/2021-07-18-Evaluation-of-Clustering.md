---
layout: post
title: "기계학습, Evaluating of Clustering"
description: "Evaluating of Clustering, 클러스터링 평가"
categories: [MachineLearning]
tags: [Machine Learning, Unsupervised Learning, Kmeans Clustering, Evaluating of Clustering]
use_math: true
redirect_from:
  - /2021/07/18/
---

* Kramdown table of contents
{:toc .toc}      


# 클러스터링 평가

## 내부 평가

> 데이터 집합을 클러스터링한 결과 그 자체를 놓고 평가하는 방식         
> 클러스터 내 높은 유사도(high intra-cluster similarity), 클러스터 간 낮은 유사도(low inter-cluster similarity)를 가진 결과물에 높은 점수를 주는 방식         
> 오로지 클러스터링 결과물만을 보고 판단하기에, 평가 점수가 높다고 실제 참값에 가깝다는 것을 반드시 보장하지는 않는다          

### Davies-Bouldin index

@
 DB = \frac {1}{n} \sum _ {i=1}^{n} \max _ {j\neq i} \left( {\frac {\sigma _ {i}+\sigma _ {j}}{d(c_{i},c_{j})}} \right)
@

> 낮은 Davies-Bouldin index 값 = 높은 클러스터 내 유사도 + 낮은 클러스터 간 유사도 = 좋은 클러스터링   

$n$ : 클러스터의 개수      
$c_ x$ : 클러스터 $x$의 중심점         
$\sigma_ x$ : 클러스터 $x$의 데이터들과 중심점 까지의 거리의 평균값, 낮을수록 클러스터 내 유사도가 높음(밀집)         
$d(c_ i, c_ j)$ : 중심점 $c_ i, c_ j$간의 거리, 멀수록 클러스터 간 유사도가 낮음        

### Dunn index

@
D = \frac {\min _ {1\leq i<j\leq n} d(i,j) }{ \max _ {1\leq k\leq n} d^{\prime }(k)}
@

> 밀도가 높고 잘 나뉜 클러스터링 결과를 목표로 함    
> 클러스터간 최소 거리와 클러스터간 최대 거리의 비율로 정의
> 높은 Dunn index 값 = 높은 클러스터 내 유사도 + 낮은 클러스터 간 유사도  = 좋은 클러스터링


$d(i,j)$ : 클러스터 i, j 간의 거리(클러스터 간 유사도), 중심간의 거리 혹은 데이터간 거리의 평균 등으로 계산     
$d^{\prime }(k)$ : 클러스터 k의 클러스터 내 거리(클러스터 내 유사도), 클러스터 내 가장 멀리 떨어진 데이터 오브젝트 간 거리 등으로 계산           


### 실루엣 기법


@
s(i) = {\frac {b(i) - a(i)}{\max ( a(i), b(i) ) }}
@


$a(i)$ : 해당 데이터가 속한 클러스터 내부의 데이터들과의 부동성
$b(i)$ : 다른 클러스터의 데이터들과의 부동성

$ -1 \leq s(i) \leq  1 $, -1에 가까울 수록 잘못 분류되었으며, 1에 가까울수록 잘 분류된 클러스터


## 외부 평가

> 클러스터링 결과를 모범답안 혹은 외부 벤치마크 평가 기준 등을 이용해서 클러스터링 알고리즘의 정확도를 평가하는 것              
> 답지와 비교하여 채점하는 방식               

~~~ python
df = pd.DataFrame({'labels': library_labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
~~~

![image](https://user-images.githubusercontent.com/32366711/125423582-0744284d-f50b-4b6e-a476-d0632b653f5e.png)



| | 판정 | 실제 |
|:----:|:----:|:----:|
|TP|참|참|
|TN|거짓|거짓|
|FP|참|거짓|
|FN|거짓|참|

### 랜드 측정

@
RI = {\frac {TP+TN}{TP+FP+FN+TN}}
@

> 정답 / 시도 의 비율         
> 그러나 FP와 FN을 같은 비중으로 계산하기에, 이는 알고리즘을 평가하는데 좋지 않을 수 있음        


### F 측정

@
F_ \beta = \frac{(\beta^2 + 1) \cdot  P \cdot  R}{\beta^2 \cdot  P + R}
@

> 랜드 측정을 개선한 방식       
> $\beta$값을 바꿔 재현율을 조정         

정밀도 : $P =  {\frac {TP}{TP+FP}}$         
재현율 : $R =  {\frac {TP}{TP+FN}}$           

$\beta = 0$일 때, $F_ 0 = P$, $\beta$값이 커질수록 최종 F-measure에 애현율이 미치는 영향이 커짐


### 자카드 지수

@
J(A, B) = \frac{\left | A \cap B \right |}{\left | A \cup B \right |} = \frac{TP}{TP + FP + FN}
@


> 두 데이터 집합 간의 유사도를 졍량화하는 데 사용되며 0과 1사이의 값을 가짐           
> 1은 두 데이터 집합이 동일하며, 0은 공통된 요소를 전혀 가지지 않는 다는 것을 의미함       
> 두 데이터 집합 간의 공통 원소들의 개수를 두어 두 데이터 집합의 합집합의 원소로 개수를 나눈 것           

