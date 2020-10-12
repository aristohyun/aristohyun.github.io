---
layout: post
title: "Python, 자료형과 제어문"
description: "Basic"
categories: [Python]
tags: [basic]
redirect_from:
  - /2020/10/06/
---

* Kramdown table of contents
{:toc .toc}

### 리스트
<span class="margin">len() 내장함수를 통해 길이 출력</span>
<span class="margin">음수 인덱스 사용가능</span>
<span class="margin">+로 이어 붙이기 가능</span>
<span class="margin">append() 메소드로 추가 가능</span>
<span class="margin">리스트는 가변, 문자열은 불변</span>

~~~ python
arr = [1,2,3,4,5]
print(arr[0])   # 1
print(arr[-1])  # 5
print(len(arr)) # 5
~~~

### 문자열
> 파이썬 문자열은 변경X. 불변!
> 다른 문자열이 필요하면 새로 만들어야함
> '' "" 뭘쓰던 상관x

<span class="margin"> 첫 따옴표 앞에 r을 붙이면 그대로 출력가능 (raw string) </span>    
<span class="margin">len() 내장함수를 통해 길이 출력</span>

~~~ python

print('C:\user\documnet')  # Error '\d'
print(r'C:\user\documnet') # C:user\document
str = r'C:\user\documnet'
print(str)                 # C:user\document

~~~

<span class="margin">여러줄로 확장 가능</span>    

~~~ python
print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")
Usage: thingy [OPTIONS]
-h                        Display this usage message
-H hostname               Hostname to connect to

# 첫번째 줄에 \를 입력 안하면 첫 줄이 띄워져 나옴
~~~

<span class="margin">+로 이어 붙이고 * 로 반복 할 수 있음</span>
<span class="margin">문자열이 연속해서 나타나면 자동으로 합쳐짐</span>

~~~ python
>>> 3 * 'un' + 'ium' 
'unununium'

>>> text = ('Put several strings within parentheses '
...         'to have them joined together.')
>>> text
'Put several strings within parentheses to have them joined together.'
# 두 문자열 리터럴만 가능, 변수X
# 변수에서는 꼭 + 사용
~~~

#### 문자열 인덱스
<span class="margin">음수도 가능!</span>

~~~ python
>>> word = 'Python'
>>> word[0]  # character in position 0
'P'
>>> word[5]  # character in position 5
'n'
>>> word[-1]  # last character
'n'
>>> word[-2]  # second-last character
'o'
>>> word[-6]
'P'
~~~
<span class="margin">슬라이싱도 가능</span>

~~~ python
>>> word[0:2]  # characters from position 0 (included) to 2 (excluded)
'Py'
>>> word[2:5]  # characters from position 2 (included) to 5 (excluded)
'tho'
>>> word[:2] + word[2:]
'Python'
>>> word[:4] + word[4:]
'Python'
~~~

#### 문자열 포맷팅
~~~ python
"i eat %d apples." % 3      # 정수
"i eat %f apples." % 3.5   # 유리수
"i eat %s apples." % five   # 문자열
"i eat %d %s." % (3,"apples")   # 문자열
"it's %d%%" % 100   # it's 100%
"%10.4f" % 21.1234567   #           21.1234

# format 함수
"I eat {0} apples".format(3)
"I eat {0} {1}".format(3,"apples")
"I eat {num} {name}".format(num=3,name="apples")

"{0:<10}".format("hi")  # 'hi        '
"{0:>10}".format("hi")  # '        hi'
"{0:^10}".format("hi")  # '    hi    '
"{0:=^10}".format("hi") # '====hi===='

name = '박시현'
age = 24
f'나의 이름은 {박시현}입니다. 나이는 {24}입니다.'
# '나의 이름은 홍길동입니다. 나이는 30입니다.'
~~~

#### 관련 함수

<span class="margin"> `s.count(c)` 문자의 갯수 반환 </span>    
<span class="margin"> `s.find(c)` 문자가 처음으로 나온 위치 반환, 없으면 -1 </span>    
<span class="margin"> `s.index(c)` 문자가 처음으로 나온 위치 반환, 없으면 에러 </span>    
<span class="margin"> `s.join(s)` 문자열 s 사이사이에 문자 c 삽입 </span>    
<span class="margin"> `s.upper() s.lower()` 문자열 s를 모두 대문자로 바꾸거나 소문자로 바꿈 </span>    
<span class="margin"> `s.lstrip()` `s.rstrip()` `s.strip()` 왼쪽, 오른쪽, 양쪽 공백 지우기 </span>    
<span class="margin"> `s.replace(before, after)` 문자열 s의 before을 after로 바꿈</span>    
<span class="margin"> `s.split()` 아무값도 넣지 않아주면 공백을 기준으로 문자열을 나누어 리스트로 저장</span>
<span class="margin">     특정 값을 넣으면 해당 값으로 나눔</span>    

### IF
~~~ python
x = 1
if x < 0:
  print("less than zero")
elif x == 0:
  print("zero")
elif x > 0:
  print("more than zero")
else: print("this isnt num")
~~~
### FOR
