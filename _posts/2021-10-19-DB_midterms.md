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
![image](https://user-images.githubusercontent.com/32366711/138294032-a3399c71-60eb-4e7b-9b50-4135ad1e53d7.png)

   4) Many-to-many relationship 
![image](https://user-images.githubusercontent.com/32366711/138294056-46b8da4b-8581-46d3-a746-7555b3a7b5d5.png)

   5) 1-to-1 relationship  
![image](https://user-images.githubusercontent.com/32366711/138294171-c89ff5d5-bd98-4720-9022-6d03b1571a3a.png)

   6) n-ary relationship
![image](https://user-images.githubusercontent.com/32366711/138294157-0d349772-2ca6-476a-bf64-1adab385e984.png)

 ![image](https://user-images.githubusercontent.com/32366711/138294143-1d3545df-77c5-4978-a100-10fdf2498d2e.png)


# Chapter 8: Write the result for the following relational algebra expressions in the given relations. 

   1) σ salary > 20000 (instructor)

   2) σ dept_name=“Physics” (instructor)

   3) ∏ID, name, salary (instructor) or 

   4) ∏name(σ dept_name =“Physics”  (instructor))

   5) r  ∪ s, r ∩ s, r - s 

   6) ∏course_id (σ semester=“Fall”  Λ year=2017 (section))  ∪ 
       ∏course_id (σ semester=“Spring”  Λ year=2018 (section)) 

   7) ∏course_id (σ semester=“Fall”  Λ year=2017 (section)) ∩
       ∏course_id (σ semester=“Spring”  Λ year=2018 (section)) 

   8) ∏course_id (σ semester=“Fall”  Λ year=2017 (section))  − 
       ∏course_id (σ semester=“Spring”  Λ year=2018 (section))  

   9) instructor  X  teaches

   10) σ instructor.id =  teaches.id  (instructor  x teaches )) 

   11) instructor ⋈_ Instructor.id = teaches.id teaches 

   12) Physics ← σ dept_name=“Physics” (instructor)

         Music ← σ dept_name=“Music” (instructor)

        Physics  ∪ Music 

   13) σ dept_name=“Physics” (σ salary > 90.000 (instructor))  

   14) (σdept_name=“Physics” (instructor)) ⋈_ instructor.ID = teaches.ID teaches 

   15) Self join example   


# Chapter 8: Write the Relational Algebra expression for the following queries. 

   1) Select those tuples of the instructor  relation where the instructor is in the “Physics” department.

   2) Find the instructors in Physics with a salary greater $90,000 

   3) Find the names of all instructors in the Physics department. 

   4) Find the department names of all instructors, and remove duplicates. 

   5) Find all courses taught in the Fall 2017 semester, or in the Spring 2018 semester, or in both. 

   6) Find the set of all courses taught in both the Fall 2017 and the Spring 2018 semesters. 

   7) Find all courses taught in the Fall 2017 semester, but not in the Spring 2018 semester. 

   8) Find all instructors in the “Physics” and “Music” department.  

   9) Get only those tuples of  “instructor  X  teaches “ that pertain to instructors and the courses that they taught. 

   11) Find the information about courses taught by instructors in the Physics department with salary greater than 90,000. 

   12) Find the information about courses taught by instructors in the Physics department. 

   13) Find the name of instructor names that is the same address of a given instructor's name 'John'.  (Self Join)

   14) Find the names of all instructors who have a higher salary than some instructor in 'Comp. Sci'. (Self Join)

   15) Find all instructors in Comp. Sci. dept with salary > 70000. 

   16) Find the Cartesian product instructor X teaches. 

   17) Find the names of all instructors in the Art  department who have taught some course and the course_id. 

   18) Find the supervisor of the supervisor of “Bob”.  

   19) Find courses that ran in Fall 2017 or in Spring 2018. // Union

    20) Find courses that ran in Fall 2017 and in Spring 2018. // Intersect

    21) Find courses that ran in Fall 2017 but not in Spring 2018. // Minus  

    22) Find the average salary of instructors in each department. // Group by 

    23) Find the names and average salaries of all departments whose average salary is greater than 42000. // group-by and having


# Chapter 8: Write the tuple relational calculus expressions for the queries above: 1) - 23). 

# 주어진 릴레이션에 대해 정규화 문제

   - 주어진 릴레이션은 제3정규형인가? 그렇지 않으면 이에 맞게 정규화하시오.

   - 주어진 릴레이션은 BCNF 정규형인가?  그렇지 않으면 이에 맞게 정규화하시오.

   - 주어진 릴레이션은 제4정규형인가?  그렇지 않으면 이에 맞게 정규화하시오. 
