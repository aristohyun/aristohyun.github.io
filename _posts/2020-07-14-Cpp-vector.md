---
layout: post
title: "C++ vector 사용법"
decription: "C++ STL : vector"
categories: [C++]
tags: [STL, vector]
redirect_from:
  - /2020/07/14/
---
  <style>
    .small{
      font-size:12px;
    }
    .normal{
      font-size:12px;
      margin-left:15px;
    }
  </style>
* Kramdown table of contents
{:toc .toc}

### 생성

-----------------
-----------------

* `vector<int> v`   <span class="normal">  빈 컨테이너 생성</span><br>
* `vector<int> v(n)`  <span class="normal"> 기본값(0)으로 초기화된 n개의 원소를 가진 컨테이너 </span><br>
* `vector<int> v(n,x)`  <span class="normal"> x로 초기화된 n개의 원소를 가진 컨테이너 </span><br>
* `vector<int> v2(v1)`  <span class="normal"> v1을 복사한 v2  </span><br>

<br>

### 멤버 함수

--------------
--------------

* `v.assign(n,x)` <span class="normal">v에 x의 값으로 n개만큼 할당</span> 
* `v.front()` `v.back()` <span class="normal">첫번째, 마지막 원소값 리턴</span> <br>
* `v.at(i)` `v[i]` <span class="normal">i번 원소 참조</span> 
  - <span class="small">속도 : at(i) < [i]</span>
  - <span class="small">안정성 : at(i) > [i] </span>
  - <span class="small">at()은 범위를 점검하기에 느리지만 안전함</span>

----------------

* `v.push_back(x)` `v.pop_back()` <span class="normal">원소삽입, 삭제</span>
* `v.insert(i,x)` <span class="normal">i번째 위치에 x 삽입, 해당 iter 리턴</span>
* `v2.swap(v1)` <span class="normal">v2와 v1의 원소와 capacity까지 교환</span>
  - <span class="small">v1의 capacity를 없애고 싶을때도 사용 가능</span> 

---------------
* `v.erase(iter)` <span class="normal">iter이 가리키는 원소 제거</span>
  - `v.erase(start, end)` <span class="normal"> [start, end) 제거</span>

* `v.size()`   <span class="normal">원소의 갯수(int) 리턴</span>
* `v.capacity()` <span class="normal">vector의 크기 리턴</span>

* `v.clear()`  <span class="normal">모든 원소 제거 , 메모리 공간은 그대로</span> 
* `v.empty()` <span class="normal"> 비었는지 확인, true == empty</span> 

----------------

* `vector<int> ::iterator iter`
* `v.begin()` `v.end()` <span class="normal">첫번째 원소와 마지막다음 원소를 가르킴</span> 
* `v.rbegin()` `v.rend()`<span class="normal"> 역순으로 첫번째, 마지막다음 원소를 가르킴 </span>
   ~~~ c++
    vector<int> v;
    vector<int> ::iterator iter;
    for(iter=v.begin();iter!=end();iter++)
      cout << *iter << endl;
  ~~~

-----------------

* `v.reserve(n)` <span class="normal"> n개의 원소를 저장할 공간 예약, capacity 설정 </span>
* `v.resize(n)` `v.resize(n,x)` <span class="normal"> v의 크기를 n으로 변경, 더 커지는 공간은 기본값 / x로 초기화 </span><br>