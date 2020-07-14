---
layout: post
title: "C++ vector 사용법"
decription: "C++ STL : vector"
categories: [C++]
tags: [STL, vector]
redirect_from:
  - /2020/07/14/
---


# C++ vector 예제

* Kramdown table of contents
{:toc .toc}

### 생성

* ` vector<int> v `  	- 빈 컨테이너 생성
* ` vector<int> v(n) `	- 기본값(0)으로 초기화된 n개의 원소를 가진 컨테이너
* ` vector<int> v(n,x) ` - x로 초기화된 n개의 원소를 가진 컨테이너
* ` vector<int> v2(v1) ` - v1을 복사한 v2


### 멤버 함수
* ` push_back(element) ` vector 
* ` pop() `