---
layout: post
title: "Kramdown 사용법"
description: "A quick post to some kramdown features."
categories: [DevEnviroment]
tags: [gitblog]
redirect_from:
  - /2020/05/01/
---

> This is [kramdown](https://kramdown.gettalong.org/) formatting test page for [Simple Texture](https://github.com/yizeng/jekyll-theme-simple-texture) theme.

* Kramdown table of contents
{:toc .toc}


**참고사이트**
> [취미로 시작하는 웹코딩](https://kilbong0508.tistory.com/303)        
> [HEROPY Tech](https://heropy.blog/2017/09/30/markdown/)                    
> [G.J Choi](http://gjchoi.github.io/env/Kramdown(%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4)-%EC%82%AC%EC%9A%A9%EB%B2%95/)    
> [잇창명](https://eatchangmyeong.github.io/syntax/#fnref:fnrepeat:1)    

# 일반적으로

[링크](https://aristohyun.github.io "깃헙 블로그")는 `[이렇게](https://aristohyun.github.io "깃헙 블로그")` 타이틀도 띄울 수 있음    
[링크](https://aristohyun.github.io "깃헙 블로그"){: target="_blank"}는 `[이렇게](https://aristohyun.github.io "깃헙 블로그"){: target="_blank"}` 새탭에서 여는건 이렇게    
**강조는** `**이렇게**`

_기울임은_ `_이렇게_`

**_둘다 섞어서도 됨_**  `**_이렇게_**`, `***이거도됨***`

각주[^1] `[^1]`.

태그 사용 가능 `<kbd>, <code>`

<ins>밑줄</ins>과 <strike>줄긋기</strike>는 이걸로 `<ins> , <strike>`

# 인용문

> 이게 인용문인데
>
> > 이중으로도 사용 가능
>

# 리스트

* `* + - ` 이걸로 리스트 하는데
  + 탭하고 적으면 이렇게도
    - 됨
* 리스트임

1. 번호도 가능함
2. 번호는 탭한다고 이중으로 안됨


# 수평선
`* * * --- ---------- _ _ _`
* * *

---

_  _  _  _

---------------

# 이미지

`![이미지](https://kramdown.gettalong.org/overview.png)`

<a class="post-image" href="https://kramdown.gettalong.org/overview.png">
<img itemprop="image" data-src="https://kramdown.gettalong.org/overview.png" src="/assets/javascripts/unveil/loader.gif" alt="Kramdown Overview" />
</a>


# 테이블

[Table Generator](https://www.tablesgenerator.com/markdown_tables){: target="_ blank"}

# 수식    
[laTeX](https://www.codecogs.com/latex/eqneditor.php){: target="_ blank"}   

[laTeX](https://docs.latexbase.com/symbols/){: target="_ blank"}  

![image](https://user-images.githubusercontent.com/32366711/125173354-d6b19280-e1f9-11eb-8e36-9a3fff2564fe.png)


[^1]: This is a footnote.
