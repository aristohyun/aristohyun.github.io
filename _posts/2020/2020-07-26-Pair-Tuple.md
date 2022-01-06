---
layout: post
title: "C++, Pair, Tuple"
description: "STL, Standard Template Library"
categories: [C++]
tags: [Data Structure, STL, pair, tuple]
redirect_from:
  - /2020/07/26/
---
* Kramdown table of contents
{:toc .toc}
  

-------------------

> pair, tuple 모두 정렬을 하면    
> first 우선순위로 정렬 하다가 first가 같으면 second에 따라 정렬함



## Pair
> 2개의 자료형을 묶어서 사용가능케 해줌     

### 사용법    
~~~ c++
#include <utility>
typedef pair<int,int> pii;
// typedef를 사용해 이름을 재정의 할 때는 보통 자료형에 따라 함
// pair 가 int int 면 pii, int char 면 pic 등등

pair<int,int> p = make_pair(0,1);
cout << p.first << ", " << p.second;

vector<pair<int,int>> temp;
temp.push_back(make_pair(1,1)); 

vector<pii> temp;
temp.push_back(pii(1,1)); 
~~~


## Tuple    
> 3개 이상의 자료형을 묶어서 사용    
> 다양한 값을 한번에 사용 가능    
> struct보다 단순. 단순히 값만 묶어 주기 때문   


### 사용법    
~~~ c++
#include <tuple> 
typedef tuple<int,double,char> tidc;
// typedef를 사용해 이름을 재정의 할 때는 보통 자료형에 따라 함
// tuple이 int int int 면 tiii, int double string이면 tids 등등

tuple<int,int,int> t = make_tuple(0,1,2);
cout << get<0>(t) << ", " << get<1>(t) << ", "<< get<2>(t);
//pair와 달리 first, second 가 아닌 get<i>로 값을 뽑아옴

vector<tuple<int,int,int>> temp;
temp.push_back(make_tuple(1,1,1)); 

vector<tiii> temp;
temp.push_back(tiii(1,1.3,'a')); 
~~~

