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

### 표준 라이브러리
~~~ python
import math
n = math.sqrt(9,0)
print(n)
~~~

### PEP
> PEP란, Python Enhancement Proposals의 약자로서 파이썬을 개선하기 위한 제안서를 의미함.
> 1. 파이썬에 새로운 기능(Feature)을 추가하거나 구현 방식을 제안하는 Standard Track PEP
> 2. 파이썬 디자인 이슈를 설명하거나 일반적인 가이드라인 혹은 정보를 커뮤니티에 제공하는 Informational PEP
> 3. 파이썬을 둘러싼 프로세스를 설명하거나 프로세스 개선을 제안하는 Process PEP. 예를 들어, 프로세스 절차, 가이드라인, 의사결정 방식의 개선, 파이썬 개발 도구 및 환경의 변경 등등.
[PEP 8](https://www.python.org/dev/peps/pep-0008)