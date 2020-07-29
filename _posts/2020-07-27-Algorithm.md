---
layout: post
title: "C++, Algorithm"
description: "STL, Standard Template Library"
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
> 알고리즘 헤더에서 자주 쓰는 함수들    

--------------------------------------------

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

--------------------------------------------------

### find    
> 가장 먼저 찾은 k값의 itr return, itr-vec.begin() 하면 index값    
> 못찾으면 return vec.end();

~~~ c++
find(vec.begin(), vec.end(), k);
find_if(vec.begin(), vec.end(), [](int i) { return i % 3 == 2; });  //람다함수를 이용해 find_if 사용

//find는 처음 찾으면 끝나기 때문에
vector<int>::iterator current = vec.begin();
while(true){
  current = find(current, vec.end(), 1);
  if(current == vec.end()) break;
  else {
    cout << current-vec.begin() <<endl;
    current++;
  }
}//등으로 사용 가능
~~~

------------------------------------------------------------

### any_of, all_of

> any_of : 어떤 요소라도 만족하면 true  //OR    
> all_of : 모든 요소가 만족하면 true    //AND    

~~~ c++
any_of(vec.begin(), vec.end(), [](int i) { return i < 10; });
all_of(vec.begin(), vec.end(), [](int i) { return i < 10; });
~~~

--------------------------------------------------------------

### remove
> remove함수는 해당 원소들을 모두 찾아서 끝으로 모은다   

<span class="margin">대부분 컨테이너는 erase 함수가 포함되어 있다</span>    
<span class="margin">그러나 erase는 특정 원소를 찾아서 제거하는게 아닌 , 일정 범위를 제거 하는 형식</span>    
<span class="margin">그래서 remove 함수로 끝에 모은 뒤에 erase로 제거한다</span>    

~~~ c++
//remove(vec.begin(), vec.end(), k) : 모든 k를 맨 뒤로 보낸다 그리고 그 첫번째 itr return
vec.erase(remove(vec.begin(), vec.end(), k), vec.end());  //그 itr부터 end까지는 k가 모여있을 테니 모두 삭제

struct is_odd { //홀수인 모든 수 
  bool operator()(const int& i) { return i % 2 == 1; }
};
vec.erase(remove_if(vec.begin(), vec.end(), is_odd()), vec.end()); // remove_if는 조건을 받음
vec.erase(remove_if(vec.begin(), vec.end(), [](int i) -> bool { return i % 2 == 1; }), vec.end());
//[](int i) -> bool { return i % 2 == 1; } : 람다 함수, lamda function, 익명 함수
~~~

---------------------------------------------------------

### transform
~~~ c++
transform(vec.begin(), vec.end(), vec.begin(), [](int i) { return i + 1; }); //배열의 모든 수에 +1씩
//vec.begin() 부터 vec.end()까지의 내역에 +1 씩 한 것을 vec.begin()부터 저장한다
~~~
