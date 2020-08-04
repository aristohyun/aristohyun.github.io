---
layout: post
title: "작성중 : 최단거리 알고리즘"
description: "Dijkstra, Prim"
categories: [Algorithm]
tags: [탐색, Dijkstra, Prim]
redirect_from:
  - /2020/07/30/
---

* Kramdown table of contents
{:toc .toc}

[원본 사이트](https://mattlee.tistory.com/50)

## 최단 경로    
> 정점 u와 정점 v를 연결하는 경로 중 간선들의 **가중치 합이 최소**가 되는 경로를 찾는 문제     
> 간선의 가중치는 경우에 따라 비용, 거리, 시간 등으로 해석될 수 있다    
    
<span class="margin">가중치는 가중치 인접 행렬이라고 불리는 2차원 배열에 저장된다</span>    
<span class="margin">그러나 가중치 인접 행렬에서는 간선의 가중치 자체가 0일 수도 있기 때문에 간선이 없음을 나타낼 때 0이라는 값을 사용할 수가 없다</span>    
<span class="margin">따라서 무한대와 같은 값을 입력해 '사실상 도달할 수 없다' 라는 뜻으로 사용한다</span>    


##  Dijkstra의 최단 경로 알고리즘

> 하나의 시작 정점으로부터 모든 다른 정점까지의 최단 경로를 찾는 알고리즘    
> Dijkstra의 알고리즘에서는 시작 정점에서 집합 S에 있는 정점만을 거쳐서 다른 정점으로 가는 `최단 거리를 기록하는 배열`이 반드시 있어야 한다

#### 핵심 알고리즘
1. <span class="margin"> 알고리즘의 매 단계에서 집합 S 안에 있지 않은 정점 중에서 가장 distance 값이 작은 정점을 S에 추가한다. </span>    
2. <span class="margin"> 새로운 정점 u가 S에 추가되면, S에 있지 않은 다른 정점들의 distance 값을 수정한다. </span>    
3. <span class="margin"> 시작 기준점이 u로 바뀌었기 때문에, 새로 추가된 정점 u를 거쳐서 정점까지 가는 거리와 기존의 거리를 비교한다.  </span>    
4. <span class="margin"> 그 후 더 작은 거리값을 기준으로 distance값을 수정한다.</span>    

~~~ c++
 distance[w] = min(distance[w], distance[u] + weight[u][w]) //현재까지 w에 도달하는 가장 짧은 거리, u에서 w까지 가는 가장 거리 중 최소치
~~~

