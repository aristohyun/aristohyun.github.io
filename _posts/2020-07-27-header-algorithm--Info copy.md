---
layout: post
title: "C++ algorithm"
description: "알고리즘에서 자주 쓰는 함수들"
categories: [C++]
tags: [STL algorithm]
redirect_from:
  - /2020/07/27/
---
  <style>
    .margin {
      font-size:12px;
      margin-left:12px;
    }
    .nomargin{
      font-size:12px;
      margin-left:0;
    }
    .space{
      margin:-10px 0;
    }
  </style>

* Kramdown table of contents
{:toc .toc}

####  #include
`#include <algirithm>`


### sort    
<span class="margin">vector 와 deque만 가능</span>   
<span class="margin">: 임의 접근 반복자(RandomAccessIterator) 타입을 만족해야 한다</span>     

~~~ c++
struct int_compare {  //원리
  bool operator()(const int& a, const int& b) const { return a > b; }
};

sort(arr.begin(),arr.end());    //오름차순 : defalt 
sort(vec.begin(), vec.end(), int_compare());   //내림차순 (핸드메이드)
sort(vec.begin(), vec.end(), greater<int>());  //내림차순 (템플릿)
~~~

<span class="margin">일부만 정렬, [start,end) 범위에서(배열 전체 범위에서) [start,middle) 까지만 정렬</span>     
<span class="margin">ex) 100명의 학생 중 상위 10명의 정보만 필요할 때</span>     

~~~ c++
partial_sort(start, middle, end);
partial_sort(vec.begin(), vec.begin()+10, vec.end());
~~~

### find

#### 