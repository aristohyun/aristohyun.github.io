---
layout: post
title: "Kaggle, Iris Classification"
description: "KMEANS, KMeans Clustering"
categories: [MachineLearning]
tags: [Kaggle, Machine Learning, Unsupervised Learning, Kmeans Clustering, Iris Dataset]
use_math: true
redirect_from:
  - /2021/07/12/
---

* Kramdown table of contents
{:toc .toc}           

# Iris Dataset

[참고 : Iris Dataset](https://eunsukimme.github.io/ml/2019/12/16/K-Means/){:target="_ blank"}         
[참고 : Iris Dataset](https://eunsukimme.github.io/ml/2019/12/16/K-Means/){:target="_ blank"}         

## 데이터셋 설명 및 분포

x : 꽃받침의 길이    
y : 꽃받침의 넓이    

길이와 넓이로 어떤 붓꽃인 분류하려고 함    

~~~ python
from matplotlib import pyplot as plt
x = samples[:, 0]
y = samples[:, 1]
plt.scatter(x, y, alpha=0.5)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()
~~~

![image](https://user-images.githubusercontent.com/32366711/125310701-cb7f7380-e36d-11eb-8ded-6361d0f7d2c2.png)

## 파이썬 코드

### 초기 중심 설정

#### 무작위 분할

~~~ python
def RandomPartition():
    labels = np.array([random.choice(range(k)) for _ in range(len(samples))])
    
    for i in range(k):
        points = [ sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i ]
        # points의 각 feature, 즉 각 좌표의 평균 지점을 centroid로 지정
        centroids[i] = np.mean(points, axis=0)
    
    plt.scatter(x, y, c=labels, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()
    
    label = makeCluster(centroids)
    
    plt.scatter(x, y, c=label, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()
    
    return label, centroids
~~~

![image](https://user-images.githubusercontent.com/32366711/125311644-8445b280-e36e-11eb-9635-a95b91de147c.png)


![image](https://user-images.githubusercontent.com/32366711/125311556-755f0000-e36e-11eb-9692-d0019f2fabd6.png)


#### Forgy

~~~ python
def Forgy():
    # 주어진 데이터 중 k개를 선택
    labels = np.zeros(len(samples))
    cen_index = random.sample(range(len(samples)), 3)
    centroids = np.array([[samples[i][0],samples[i][1]] for i in cen_index])

    labels = np.zeros(len(samples))
    # 각 데이터를 순회하면서 centroids와의 거리를 측정합니다
    for i in range(len(samples)):
        distances = np.zeros(k)	# 초기 거리는 모두 0으로 초기화 해줍니다
        for j in range(k):
            distances[j] = distance(sepal_length_width[i], centroids[j])
            cluster = np.argmin(distances)	# np.argmin은 가장 작은 값의 index를 반환합니다
            labels[i] = cluster
    
    plt.scatter(x, y, c=labels, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()
    
    return labels, centroids
~~~

![image](https://user-images.githubusercontent.com/32366711/125311525-6f691f00-e36e-11eb-8dd0-47fe078a3950.png)


#### MacQueen

~~~ python
def MacQueen():
    # 주어진 데이터에서 k개 선택 ( Forgy와 동일 )
    # but, 다른 데이터들을 클러스터에 추가할 때 마다 초기 mu를 재설정함
    clusters = [[],[],[]]
    labels = np.zeros(len(samples))
    cen_index = random.sample(range(len(samples)), 3)
    centroids = np.array([[samples[i][0],samples[i][1]] for i in cen_index])

    for i in range(len(samples)):
        distances = np.zeros(k)	# 초기 거리는 모두 0으로 초기화 해줍니다
        for j in range(k):
            distances[j] = distance(sepal_length_width[i], centroids[j])
        cluster = np.argmin(distances)	# np.argmin은 가장 작은 값의 index를 반환합니다
        labels[i] = cluster
        clusters[cluster].append(sepal_length_width[i])
        centroids[cluster] = np.mean(clusters[cluster], axis=0)

    plt.scatter(x, y, c=labels, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()
    
    return labels, centroids
~~~
![image](https://user-images.githubusercontent.com/32366711/125311502-6aa46b00-e36e-11eb-88a9-14d203784ca6.png)



#### Kaufman

~~~ python
def nearest_centroid(x_y, centroids):
    distances = []
    for i in range(k):
        if centroids[i][0] == -1 and centroids[i][1] == -1:
            break
        distances.append(distance(x_y, centroids[i]))
    cluster = np.argmin(distances)	# np.argmin은 가장 작은 값의 index를 반환합니다
    return centroids[cluster]
    
def Kaufman():
    # 데이터 집합 중 가장 중심에 위치한 데이터를 첫번째 중심으로 설정한다. 
    # 이후 선택되지 않은 각 데이터들에 대해, 가장 가까운 무게중심 보다 선택되지 않은 데이터 집합에 
    # 더 근접하게 위치한 데이터를 또 다른 중심으로 설정하는 것을 k번 반복
    
    distances = np.zeros(len(samples))
    centroids_index = np.array([-1 for _ in range(k)], int)
    centroids = np.array([[-1,-1] for _ in range(k)], float)
    
    # 1. 전체 데이터의 중심을 구해서, 가장 가까운 데이터 찾기
    middle = np.mean(sepal_length_width, axis=0)
    for i in range(len(samples)):
        distances[i] = distance(sepal_length_width[i], middle)
    nearest_index = np.argmin(distances)
    centroids[0] = sepal_length_width[nearest_index]   # 초기 mu
    centroids_index[0] = nearest_index

    # 2. 다른 데이터들에 D-d 값이 가장 큰 데이터 찾기
    for n in range(1, k):
        C = np.zeros(samples.size)
        for i in range(len(samples)):
            w_i = sepal_length_width[i]
            for j in range(len(samples)):     
                w_j = sepal_length_width[j]
                D_j = distance(nearest_centroid(w_j, centroids), w_j)
                d_ji = distance(w_j, w_i)
                C[i] += max(D_j - d_ji, 0)
        # Select C max
        centroids_index[n] = np.argmax(C)
        centroids[n] = sepal_length_width[centroids_index[n]]
        
    labels = makeCluster(centroids)
        
    plt.scatter(x, y, c=labels, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()

    return labels, centroids
~~~

![image](https://user-images.githubusercontent.com/32366711/125311466-64ae8a00-e36e-11eb-944e-c8126b0c6aa1.png)


### 무게중심 재설정

~~~ python
def get_new_centroid(labels):
    new_centroids = np.array([[0,0] for _ in range(k)], float)
    for i in range(k):
      # 각 그룹에 속한 데이터들만 골라 points에 저장합니다
      points = [ sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i ]
      # points의 각 feature, 즉 각 좌표의 평균 지점을 centroid로 지정합니다
      new_centroids[i] = np.mean(points, axis=0)
    
    return new_centroids
~~~

### 클러스터 재 할당

~~~ python
def make_cluster(centroids):
    labels = np.zeros(len(samples))
    # 각 데이터를 순회하면서 centroids와의 거리를 측정합니다
    for i in range(len(samples)):
        distances = np.zeros(k)	# 초기 거리는 모두 0으로 초기화 해줍니다
        for j in range(k):
            distances[j] = distance(sepal_length_width[i], centroids[j])
        cluster = np.argmin(distances)	# np.argmin은 가장 작은 값의 index를 반환합니다
        labels[i] = cluster
    return labels
~~~

### 반복

~~~ python

label, centroids = RandomPartition()
new_centroids = get_new_centroid(labels)
new_labels = make_cluster(new_centroids)

while centroids != new_centroids:
  centroids = new_centroids
  new_centroids = get_new_centroid(labels)
  new_labels = make_cluster(new_centroids)

~~~

![image](https://user-images.githubusercontent.com/32366711/125315751-624e2f00-e372-11eb-830d-d7984e59d52d.png)

![image](https://user-images.githubusercontent.com/32366711/125315891-8873cf00-e372-11eb-95d5-9a761182592f.png)

![image](https://user-images.githubusercontent.com/32366711/125315912-8f024680-e372-11eb-80fd-8d9cdc4792bb.png)

![image](https://user-images.githubusercontent.com/32366711/125315928-932e6400-e372-11eb-9168-8e4b4e0cf63a.png)

## 라이브러리 sklean.cluster.KMeans

~~~ python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
samples = iris.data

# 3개의 그룹으로 나누는 K-Means 모델을 생성합니다
model = KMeans(n_clusters = 3)
model.fit(samples)
labels = model.predict(samples)

# 클러스터링 결과를 시각화합니다
x = samples[:, 0]
y = samples[:, 1]
plt.scatter(x, y, c=labels, alpha=0.5)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()
~~~

![image](https://user-images.githubusercontent.com/32366711/125315999-a3deda00-e372-11eb-8fe0-234ffccacbc3.png)


### 성능 확인

iris 데이터 셋은 어떤 특성인지 정답을 가지고 있기에 가능

~~~ python
species = np.chararray(target.shape, itemsize=150)
for i in range(len(samples)):
    if target[i] == 0:
        species[i] = 'setosa'
    elif target[i] == 1:
        species[i] = 'versicolor'
    elif target[i] == 2:
        species[i] = 'virginica'


df = pd.DataFrame({'labels': RP_labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

df = pd.DataFrame({'labels': Forgy_labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

df = pd.DataFrame({'labels': MQ_labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

df = pd.DataFrame({'labels': Kauf_labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

df = pd.DataFrame({'labels': library_labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
~~~


RP          

| labels \ species | setosa | versicolor | virginica |          
|:----------------:|:------:|:------------:|:-----------:|                
| 0                |    0    |    12        |     34      |          
| 1                |     0   |     38       |     15      |                    
| 2                |     50   |      0      |       1    |          


Forgy          

| labels \ species | setosa | versicolor | virginica |          
|:----------------:|:------:|:------------:|:-----------:|          
| 0                |    50    |    0        |     1      |           
| 1                |     0   |     12       |     34      |          
| 2                |     0   |      38      |       15    |          


MQ          

| labels \ species | setosa | versicolor | virginica |          
|:----------------:|:------:|:------------:|:-----------:|            
| 0                |    50    |    6        |     1      |          
| 1                |     0   |     6       |     21      |          
| 2                |     0   |      38      |       28    |          


Kauf          

| labels \\ species | setosa | versicolor | virginica |          
|:----------------:|:------:|:------------:|:-----------:|          
| 0                |    0    |    38        |     15      |          
| 1                |     50   |     0       |     1      |          
| 2                |     0   |      12      |       34    |          


library          

| labels  species | setosa | versicolor | virginica |
|:----------------:|:------:|:------------:|:-----------:|
| 0                |    50    |    0        |     0      |
| 1                |     0   |     48       |     14     |
| 2                |     0   |      2      |       36    |


1
