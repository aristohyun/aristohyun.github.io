---
layout: post
title: "#include algorithm"
description: "알고리즘에서 자주 쓰는 함수들"
categories: [C++]
tags: [STL, algorithm]
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



### remove
> remove함수는 해당 원소들을 모두 찾아서 끝으로 모은다   

<span class="margin">대부분 컨테이너는 erase 함수가 포함되어 있다</span>
<span class="margin">그러나 erase는 특정 원소를 찾아서 제거하는게 아닌 , 일정 범위를 제거 하는 형식    </span>
<span class="margin">그래서 remove 함수로 끝에 모은 뒤에 erase로 제거한다</span>

~~~ c++
remove(vec.begin(), vec.end(), k) //모든 k를 맨 뒤로 보낸다 그리고 그 첫번째 itr return
vec.erase(remove(vec.begin(), vec.end(), k), vec.end());  //그 itr부터 end까지는 k가 모여있을 테니 모두 삭제

struct is_odd {
  bool operator()(const int& i) { return i % 2 == 1; }
};
vec.erase(remove_if(vec.begin(), vec.end(), is_odd()), vec.end()); // remove_if는 조건을 받음
~~~
