---
layout: post
title: "C++, Priority Queue"
description: "STL, Standard Template Library"
categories: [C++]
tags: [Data Structure, STL, priority queue, heap]
redirect_from:
  - /2020/07/14/
---
* Kramdown table of contents
{:toc .toc}

> 우선순위 큐는 Heap의 형태    
> 우선순위가 같다면 선입선출    

-------------------

### 생성자


&nbsp;&nbsp;&nbsp;&nbsp;  `#include <queue>` <span class="nomargin">필요</span>

~~~ c++    
struct cmp{
  bool operator()(const int& a, const int& b) const { return a > b; }
};

priority_queue<int> pq; //Max Heap
priority_queue<int, vector<int>, cmp> pq  //Max Heap
priority_queue<int, vector<int>, less<int>> pq  //Max Heap, top이 최대값
priority_queue<int, vector<int>, greater<int>> pq;  //Min Heap, top이 최소값

~~~    
------------------------

### 멤버함수

* `pq.push(x)` <span class="margin"> pq에 x 삽입, 자동정렬됨</span>
* `pq.pop()` <span class="margin">top의 원소 삭제<span>
* `pq.top()` <span class="margin">top의 원소, 가장 크거나 가장 작은 원소 return<span>

* `q.empty()` <span class="margin">비었는지 확인, true == empty</span>
* `q.size()`  <span class="margin">원소 갯수 리턴, int </span>


~~~ c++
#include <queue>

using namespace std;

class Student{
private:
    string name;
    int id;
public:
    int operator()(Student a, Student b){ //비교함수자 cmp
      return a.id > b.id;
    }

    Student(){this->name=""; this->id=0;}
    Student(string _name, int _id){this->name = _name; this->id = _id;}
    void print(){
        cout << this->id <<" : "<< this->name<<endl;
    }
};

int main() {
    priority_queue<Student,vector<Student>,Student> pq;
    
    pq.push(Student("si",10));
    pq.push(Student("hyeon",13));
    pq.push(Student("park1",8));
    pq.push(Student("park2",8));
    
    cout << pq.size() << endl;
    
    while(!pq.empty()){
        Student now = pq.top();
        pq.pop();
        now.print();
    }
    return 0;
}
~~~
