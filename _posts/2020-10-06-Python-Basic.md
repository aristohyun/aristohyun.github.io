---
layout: post
title: "Python, 기초"
description: "Basic"
categories: [Python]
tags: [basic]
redirect_from:
  - /2020/10/06/
---

* Kramdown table of contents
{:toc .toc}

## 기본사용법
<span class="margin">중괄호X -> 콜론 + 들여쓰기(동일한 블럭은 동일한 수의 공백)</span>

~~~ python
if x > 0:
    a = 1
    b = 2
    c = a + b
else :
    a = -1
    b = -2
    c = a - b
~~~

### 사용자 입출력
~~~ python
str = input("문자열을 입력해주세요: ")
print(str)
~~~

### 파일 입출력
#### 입력
~~~ python
f = open("test.txt", 'w') # r, w, a
for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()

with open("sample.txt", "w") as f:
    f.write("Life is too short, you need python")
# with 블록을 벗어나는 순간 close됨
~~~

#### 출력
> readline(), 한줄을 문자열로    
> readlines(), 한줄씩 리스트에 저장    
> read(), 전체 문자열을 하나의 문자열에    

~~~ python
f = open("test.txt", 'r')

line = f.readline() # 첫번째 줄만 출력
print(line)

while True: # 모든 줄 출력
    line = f.readline()
    if not line: break
    print(line)

lines = f.readlines() # 리스트로 반환, 각 라인이 리스트에 한 줄 씩
for line in lines:
    print(line)

data = f.read() # 페이지 전체의 내용이 하나의 문자열로
print(data)

f.close()
~~~
#### 추가
~~~ python
f = open("test.txt",'a')
for i in range(1, 10):  # 기존 내용의 끝에서부터 추가 
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()
~~~
## 표준 라이브러리
~~~ python
import math
n = math.sqrt(9,0)
print(n) # 주석은 샵하나, 띄어쓰기 하나 하는게 좋음
~~~

## [PEP 8](https://www.python.org/dev/peps/pep-0008){: target="_blank"}
> PEP란, Python Enhancement Proposals의 약자로서 파이썬을 개선하기 위한 제안서를 의미함.    

1. 파이썬에 새로운 기능(Feature)을 추가하거나 구현 방식을 제안하는 Standard Track PEP
2. 파이썬 디자인 이슈를 설명하거나 일반적인 가이드라인 혹은 정보를 커뮤니티에 제공하는 Informational PEP
3. 파이썬을 둘러싼 프로세스를 설명하거나 프로세스 개선을 제안하는 Process PEP. 예를 들어, 프로세스 절차, 가이드라인, 의사결정 방식의 개선, 파이썬 개발 도구 및 환경의 변경 등등.     

[PEP 8](https://www.python.org/dev/peps/pep-0008){: target="_blank"}


## 연산자

> ### 산술연산자
> `+ - * / % // ** ***`
> ### 비교연산자
> `== != < > <= >=`
> ### 할당연산자
> `=`, `+= -= *= /= %= //=`
> ### 논리연산자
> `and` (&&), `or` (||), `not` (!)    
> ### Bitwise 연산자
> `&` (and), `|` (or), `^` (xor), `~` (complement), `<<` `>>` (shift)    
> 비트단위 연산
> ### 멤버쉽 연산자
> `in`, `not in`    
> 좌측 Operand가 우측 컬렉션에 속했는지 아닌지 확인, return Bool    
> ### Identity 연산자
> `is`, `is not`    
> 양쪽 Operand가 동일한 Object를 가리키는지 아닌지를 체크, return Bool
