~~~
---
layout: post
title: "Title"
description: "description"
categories: [category]
tags: [Tag1, Tag2]
use_math: true
redirect_from:
  - /2021/12/31
---

* Kramdown table of contents
{:toc .toc} 
~~~

> This is [kramdown](https://kramdown.gettalong.org/) formatting test page for [Simple Texture](https://github.com/yizeng/jekyll-theme-simple-texture) theme.  

# 참고사이트

> [code box](https://rdmd.readme.io/docs/code-blocks)
> [취미로 시작하는 웹코딩](https://kilbong0508.tistory.com/303)        
> [HEROPY Tech](https://heropy.blog/2017/09/30/markdown/)                    
> [G.J Choi](http://gjchoi.github.io/env/Kramdown(%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4)-%EC%82%AC%EC%9A%A9%EB%B2%95/)    
> [잇창명](https://eatchangmyeong.github.io/syntax/#fnref:fnrepeat:1)    

# 리눅스 파일명 일괄 변경 방법

`rename 's/2021/2022/' 2021-01-*.txt`

# 깃

## 강제 푸시

`git push -u origin master --force`

## 리셋

`git reset --hard DI`

## 커밋 합치기

[완전범죄 영상](https://youtu.be/omXz1t-u_6k?t=2215)

1. `git log` 를 통해 마지막 commit ID를 찾은 후
2. `git rebase -i ID`를 통해 해당 위치로 이동
3. 그러면 해당 커밋 이후의 커밋 내역이 출력이 될틴데, 맨 위에 커밋 빼고 모두 pick -> squash 로 변경
4. 그러면 커밋 텍스트 변경 화면이 됨. 싹 지우고 원하는 내용 입력
5. git push -f로 올리기


# 테이블

[Table Generator](https://www.tablesgenerator.com/markdown_tables){: target="_ blank"}

# 접어두기

<details>
  <summary>Click to expand!</summary>
  
  ## Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>

# 다단나누기

~~~
<div class="multi-stage">
  <div class="stage" markdown=1>
  </div>
  <div class="stage" markdown=1>
  </div>
</div>
~~~

# 코드
## Simple Highlight

> c++ python java javascript html

`~~~ c++ ~~~`
  
`{% highlight c++ linenos=table %} {% endhighlight %}`


~~~ c++

#include <iosteam>
using namespace std;

int main(){
  cout << "hello world"<<endl;
  return 0;
}

~~~

{% highlight c++ linenos=table %}
#include <iosteam>
using namespace std;

int main(){
  cout << "hello world"<<endl;
  return 0;
}
{% endhighlight %}


# 수식    
  
[laTeX 툴](https://www.codecogs.com/latex/eqneditor.php){: target="_ blank"}   

[laTeX 공식문서](https://docs.latexbase.com/symbols/){: target="_ blank"}  


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


[^1]: This is a footnote.
