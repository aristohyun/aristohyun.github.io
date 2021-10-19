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

   - Terminolgies: Data, Database, database system, DBMS, database schema, database instance 

   - Data model concept and its categories 

   - Data modeling concepts and its categories: Conceptual modeling, Logical modeling, Physical modeling  

   - Data dependency and data independencies (logical and physical) 

   - 시스템 카탈로그 예제 보여주기 (Relation,relation's column)

   - 3-계층 스키마 아키텍처 

   - DBMS를 구성하는 주요 구성요소와 각 모듈의 기능 설명

   - 3-tier 클라이언트/서버 아키택처 


# Chapter 3-5 & Chapter 9: 주어진 요구사항에서 데이터베이스 설계: 간단한 Company 데이터베이스, 수강신청시스템, ... 

   - E-R Diagram 또는 Class Diagram 그리기

   - CRC 카드(스키마 테이블) 작성하기 

   - E-R to CRC (relational database schema table) mapping algoriths 

     1)  Weak entity type 

     2) Multi-valued attribute

     3) 1-to-many relationship 

     4) Many-to-many relationship 

     5) 1-to-1 relationship  

     6) n-ary relationship

 

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
