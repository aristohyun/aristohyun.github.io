---
layout: post
title: "DBS, Example questions for midterms"
description: "DBS, 이종연 교수님"
categories: [DBS]
tags: [2021-2, DBS, Database System, DBMS, 이종연]
use_math: true
redirect_from:
  - /2021/10/19/
---

* Kramdown table of contents
{:toc .toc}   


# Chapters 1-2:

## Terminolgies

#### Data

`data == value`        
- 측정할 수 있거나, 표현할 수 있는 값
    - 피지컬 벨류 : 물리적인 값, 키, 몸무게, 성적 드등
    - 로지컬 벨류 : 정서적 추상적 주관적인 값 (행복도)

#### Database

`Assets of Operational data`
- 단체를 운영, 관리하는데 필요한 데이터들의 집합

#### DBMS

`DB Software`
- 데이터베이스를 정의, 조작하는 기능을 가진 소프트웨어 패키지
- 나아가 DB의 모든 엑세스를 컨트롤하는 소프트웨어

유저가 운영 프로그램에서 DB를 억세스 하려면 반드시 DBMS를 거쳐야함

#### database system

- Database가 잘 동작하게 하는 환경.
- DBMS를 포함하는 모든 데이터 요소들을 포함함. 

좁은 의미로 : 데이터베이스 + DBMS
넓은 의미로 : 데이터베이스 + DBMS + 컴퓨터 + 네트워크 + 쿼리 + 프로그래머 등등의 모든 구성요소

#### Data models

`A Set of Concepts`

- 개념(툴)들의 집합
- 데이터베이스의 구조, 연산, 제약조건들의 집합
- 데이터모델이 바뀌면 구조가 바뀌고 연산이 바뀌고 제약조건이 바뀜

#### database schema

`The Description of a Database`

- 데이터베이스 구조, 타입, 제약조건을 포함하는 데이터베이스의 개념

#### database instance 

`Database Snapshot`

- 특정 시점에 데이터베이스에 저장된 실제 데이터

## Data model concept and its categories 

### Structure 

- 데이터베이스 구조 정의
- 요소 뿐 아니라, 요소 그룹및 해당 그룹간의 관계도 포함됨

### Constraintrn

- 유효한 데이터에 대한 일부 제한을 지정하며
- 이러한 제약 조건은 항상 적용되어야 함

### Categories

- Conceptual[^Conceptual] data models
    - 데이터를 인식하는 방식에 가까운 개념
    - 엔티티 기반, 객체 기반 데이터 모델
    - E-R 모델, 향상된 E-R 모델
- Implementation[^Implementation] data models
    - 상용 DBMS 구현에 사용되는 위의 두 가지 사이에 해당하는 개념
    - 관계형 데이터 모델, 객체 지향 데이터 모델(OODM)
- Physical[^Physical] data models
    - 데이터가 컴퓨터에 저장되는 방법에 대한 세부 정보를 설명
    - 관계형 데이터 모델로 구현된 데이터를 물리적으로 저장
    - 파일 기반 스토리지 구조
- Self-Describing data models
    - 데이터에 대한 설명을 데이터 값과 결합
    
[^Conceptual]: High-level, Semantic
[^Implementation]: Representational 
[^Physical]: low-level, Internal

## Data Modeling

- 데이터베이스를 설계, 구축하는 일련의 모든 과정

![image](https://user-images.githubusercontent.com/32366711/138269458-ed83084b-69b4-4593-a5da-b25667767619.png)

## 3-계층 스키마 아키텍처 

- 외부 스키마를 참조하며, 실행을 위해 DBMS에 의해 내부스키마로 매핑
- 내부 DBMS 수준에 추출된 데이터는 사용자의 외부 뷰와 일치하도록 다시 포맷됨

![image](https://user-images.githubusercontent.com/32366711/138272965-43690457-a258-4d7e-9650-5c1995fa4509.png)

## Data dependency and data independencies (logical and physical) 

### Data dependency

- 이미 수행된 데이터의 변화가 뒤의 수행 결과에 영향을 끼치는 것

### Data Independencies

- 하위 단계의 데이터 구조가 변경되더라도 상위 단계에 영향을 미치지 않는 것

- Logical Data Independence
    - Ecternal[^External] 스키마와 관련이 있음
    - 애플리케이션을 변경하지 않고, 개념적 스키마를 변경할 수 있는 것

- Physical Data Independence
    - 개념적 스키마를 변경하지 않고 Internal[^Internal] 스키마를 변경할 수 있는 것
    - 특정 파일 구조가 재구성, 성능 향상을 위한 새 인덱스 작성

[^Internal]: 물리적 스퇴지 구조 및 액세스 경로를 설명하는 내부 레벨
[^External]: 다양한 사용자 보기를 설명하는 외부 레벨

## 시스템 카탈로그 예제 보여주기 (Relation,relation's column)

- 데이터 딕셔너리
- 데이터베이스 관리자의 도구로, 모든 개체들에 대한 정의나 명세에 대한 정보가 수록되어 있는 시스템 테이블

#### RELATIONS

| Relation_name | No_of_columns |
|:-------------:|:-------------:|
| STUDENT | 4 |
| COURSE | 4 |

#### COLUMNS

| Column_name | Data_type | Belongs_to_relation |
|:-----:|:-------:|:-------:|
| Name | Character(30) | STUDENT |
| Student_number | Character(4) | STUDENT |
| Class | Integer(1) | STUDENT |
| Major | Major_type | STUDENT |
| Course_name | Charter(20) |COURSE |
| Course_number| Character(10) |COURSE |
| Credit_hours | Integer(1) |COURSE |
| Department | Charter(5) |COURSE |




## DBMS를 구성하는 주요 구성요소와 각 모듈의 기능 설명

![image](https://user-images.githubusercontent.com/32366711/138289267-8ff7fc94-38a4-49a0-bea2-3220fa7cc3cb.png)

- 질의처리기
    - DDL, DML, DCL 명령어가 들어오면 해석해서 처리하는 역할
- DML 예비 컴파일러
    - 호스트 프로그래밍 언어로 작성된 응용 프로그램 속에 삽입되어 있는 DML 명령문 추출, 함수 호출문 삽입
- DDL 컴파일러
    - 데이터 정의어로 작성된 스키마를 해석
- DML 컴파일러
    - 데이터 조작어 요청을 분석하여, 런타임 데이터 베이스 처리기가 이해할 수 있도록 해석
- 런타임 데이터베이스 처리기
    - 저장 데이터 관리자를 통해 DB에 접근하여, DML 컴파일러부터 전달 받은 요청을 DB에서 실행
- 트랜잭션 관리자
    - DB접근 과정에서 사용자 접근 권한이 유요한지 검사
    - 제약조건 위반 여부 확인
- 저장 데이터 관리자
    - 디스크에 저장되어 있는 사용자 DB와 데이터 사전을 관리, 접근


## 3-tier 클라이언트/서버 아키택처 

- Client + App <–> Web <-> DBMS
- 2티어에서 Web과 DBMS가 분리된 방식
- Web 어플리케이션이 주로 채택

- 최근의 3티어 방식
    - Client + Web Browser <-> Web Server + App(WAX) <-> DBMS


# Chapter 3-5 & Chapter 9: 주어진 요구사항에서 데이터베이스 설계: 간단한 Company 데이터베이스, 수강신청시스템, ... 

 - E-R Diagram 또는 Class Diagram 그리기

 - CRC 카드(스키마 테이블) 작성하기 

 - E-R to CRC (relational database schema table) mapping algorithms 

   1)  Weak entity type 
    - 두줄 박스
   2) Multi-valued attribute
    - 두줄 타원
   3) 1-to-many relationship 
![image](https://user-images.githubusercontent.com/32366711/138294327-de3bfee4-546a-416f-8073-86010ad47cae.png)

   4) Many-to-many relationship 
![image](https://user-images.githubusercontent.com/32366711/138294056-46b8da4b-8581-46d3-a746-7555b3a7b5d5.png)

   5) 1-to-1 relationship  
![image](https://user-images.githubusercontent.com/32366711/138294171-c89ff5d5-bd98-4720-9022-6d03b1571a3a.png)

   6) n-ary relationship
![image](https://user-images.githubusercontent.com/32366711/138294157-0d349772-2ca6-476a-bf64-1adab385e984.png)

 ![image](https://user-images.githubusercontent.com/32366711/138294143-1d3545df-77c5-4978-a100-10fdf2498d2e.png)


# Chapter 8: Write the result for the following relational algebra expressions in the given relations. 

$\sigma$ : 해당 조건을 만족하는 튜플의 모든 속성을 출력함

$\Pi$ : 추출, 해당 속성들만 출력함

$\cup , \cap , -$ : 속성타입이 같은 경우만 사용가능. 속성명이 다르면 앞에껄 사용

1) $\sigma_ \text{salary > 20000} (\text{instructor})$
  - instructor테이블에서 salary가 20000 이상인 튜플 검색

2) $\sigma_ \text{ dept\_name="Physics"} (\text{instructor})$
  - instructor테이블에서 dept_ name이 Physics 인 튜플 검색

3) $\Pi_ \text{ ID, name, salary } (\text{instructor}) $
  - instructor 테이블에서 ID, name, salary만 추출해서 출력

4) $\Pi_ \text{name}(\sigma_ \text{ dept\_name ="Physics"}  (\text{instructor})) $
  - instructor 테이블에서 dept_ name이 Physics인 튜플들의 name만 출력

5) $r \cup s, r \cap s, r - s $

6)$ \Pi_ \text{course\_id} (\sigma_ {\text{semester="Fall"} \wedge \text{year=2017}} (\text{section})) 
   \cup \Pi_ \text{course\_id} (\sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}} (\text{section})) $
  -  section 테이블에서 semester이 Fall 이고 year이 2017인 튜플과 section 테이블의 Course id와 semester이 Spring 이고 year이 2018인 튜플의 Course id의 합집합

7)$ \Pi_ \text{course\_id} (\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}} (\text{section})) 
   \cap \Pi_ \text{course\_id} (\sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}} (\text{section})) $
  -  교집합

8)$ \Pi_ \text{course\_id} (\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}} (\text{section})) 
    - \Pi_ \text{course\_id} (\sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}} (\text{section}))$  
  - 차집합

9) instructor  X  teaches
   - 곱집합, 모든 짝지을수있는 경우의 수 출력

10) $\sigma_ \text{instructor.id =  teaches.id}  (\text{instructor  X teaches } )) $
  -  instructor테이플과 teaches테이블의 튜플들의 곱집합 중에서 instructor의 id와 teaches의 id가 같은 부분 검색

11) $\text{instructor} \Join_ \text{Instructor.id = teaches.id} teaches  $
  - instructor과 teaches 테이블의 id가 같은 부분을 join, 위의 결과와 같음

12) @ \text{Physics} \leftarrow \sigma_ \text{dept\_name="Physics" (instructor)} \\\ 
         \text{Music} \leftarrow \sigma_ \text{dept\_name="Music" (instructor)} \\\ 
         \text{Physics} \cup \text{Music}  @
         
  - instructor 테이블에서 dept name이 Physics인 행렬을 검색해서 Physics라는 테이블로 저장
  - instructor 테이블에서 dept name이 Music 행렬을 검색해서 Music라는 테이블로 저장
  - Physics 테이블과 Music테이블의 합집합

13) $\sigma_ \text{dept\_name="Physics"} (\sigma_ \text{salary > 90000} (\text{instructor})) $ 
    - instructor 테이블의 salary > 90000 인 튜플 중에서 dept name="Physics"인 튜플
    - $\sigma_ {\text{dept\_name="Physics"} \wedge \text{salary > 90000}} (\text{instructor}) $
    - 가 성능이 더 좋음

14) $(\sigma_ \text{dept\_name="Physics"} (\text{instructor})) \Join_ \text{instructor.ID = teaches.ID} \text{teaches} $
    - instructor 테이블 중에서 dept name="Physics" 인 튜플들과 teaches 테이블들의 join 결합(ID를 기준으로) 

   15) Self join example   
  - instructor 테이블중에서 name="John"과 같은 dept name을 가진 테이블 검색
  - $\text{JOHN} \leftarrow \sigma_ \text{name="John"} (\text{instructor}) \\\ 
     \text{JOHN} \Join_ \text{dept\_name="Physics"} \text{instructor}
    $

# Chapter 8: Write the Relational Algebra expression for the following queries. 

1) Select those tuples of the instructor relation where the instructor is in the "Physics" department.

@
\sigma_ \text{dept\_name = "Physics"} (\text{instructor})
@

2) Find the instructors in Physics with a salary greater $90,000 

@
\sigma_ {\text{dept\_name = "Physics"} \wedge \text{salary>90000}} (\text{instructor})
@

3) Find the names of all instructors in the Physics department. 

@
\Pi_ \text{name} (\sigma_ \text{dept\_name = "Physics"} (\text{instructor}))
@

4) Find the department names of all instructors, and remove duplicates. 

@
\Pi_ \text{name}(\text{instructors})
@

5) Find all courses taught in the Fall 2017 semester, or in the Spring 2018 semester, or in both. 

@
\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}}(\text{section}) \cup \sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}}(\text{section})
@

6) Find the set of all courses taught in both the Fall 2017 and the Spring 2018 semesters. 

@
\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}}(\text{section}) \cap \sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}}(\text{section})
@

7) Find all courses taught in the Fall 2017 semester, but not in the Spring 2018 semester. 

@
\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}}(\text{section}) - \sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}}(\text{section})
@

8) Find all instructors in the "Physics" and "Music" department.  

@
\sigma_ {\text{dept\_name="Physics"} \; \vee \; \text{dept\_name="Music"}} (\text{instructors})
@

9) Get only those tuples of  "instructor  X  teaches" that pertain to instructors and the courses that they taught. 


11) Find the information about courses taught by instructors in the Physics department with salary greater than 90,000. 

12) Find the information about courses taught by instructors in the Physics department. 

13) Find the name of instructor names that is the same address of a given instructor's name 'John'.  (Self Join)

14) Find the names of all instructors who have a higher salary than some instructor in 'Comp. Sci'. (Self Join)

15) Find all instructors in Comp. Sci. dept with salary > 70000. 

16) Find the Cartesian product instructor X teaches. 

17) Find the names of all instructors in the Art  department who have taught some course and the course_id. 

18) Find the supervisor of the supervisor of “Bob”.  

19) Find courses that ran in Fall 2017 or in Spring 2018. // Union

@
\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}}(\text{courses}) \cup \sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}}(\text{courses})
@

20) Find courses that ran in Fall 2017 and in Spring 2018. // Intersect

@
\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}}(\text{courses}) \cap \sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}}(\text{courses})
@

21) Find courses that ran in Fall 2017 but not in Spring 2018. // Minus  

@
\sigma_ {\text{semester="Fall"}  \wedge \text{year=2017}}(\text{courses}) - \sigma_ {\text{semester="Spring"}  \wedge \text{year=2018}}(\text{courses})
@

22) Find the average salary of instructors in each department. // Group by 

@
{_ \text{department} \mathfrak{F} _ \txet{AVG(salary)}} \text(instructors)
@

23) Find the names and average salaries of all departments whose average salary is greater than 42000. // group-by and having


# Chapter 8: Write the tuple relational calculus expressions for the queries above: 1) - 23). 

# 주어진 릴레이션에 대해 정규화 문제


- 주어진 릴레이션은 제1정규형인가? 그렇지 않으면 이에 맞게 정규화하시오.
  - 값이 오토믹 한가
  - ex red; white; black; 처럼 한 속성에 값이 여러개인 경우 X

- 주어진 릴레이션은 제2정규형인가? 그렇지 않으면 이에 맞게 정규화하시오.
  - 완전 함수적 종속인가?
  - {SSN, Pnumber} -> {Pname, Plocation}에서 Pname과 Plocation은 SSN이 없이 Pnumber만으로 식별할 수 있다
  - {SSN, Pnumber} -> {Ename}에서 Ename은 Pnumber없이 SSN만으로도 식별이 가능하다
  - 따라서 각각 나눠서 따로 테이블을 만들어야 완전 함수적 종속이 가능해 진다

- 주어진 릴레이션은 제3정규형인가? 그렇지 않으면 이에 맞게 정규화하시오.
  - 제 3 정규형은, 이행적 종속이 없어야함. 즉 삼단논법이 안되도록 쪼개야함
  - 그런 정보는 조인해서 하면 됨

- 주어진 릴레이션은 BCNF 정규형인가? 그렇지 않으면 이에 맞게 정규화하시오.
  - A,B -> C, C-> B 면 안됨

- 주어진 릴레이션은 제4정규형인가? 그렇지 않으면 이에 맞게 정규화하시오. 
  - 다치 종속성을 제거해야함
