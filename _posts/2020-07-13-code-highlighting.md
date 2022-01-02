---
layout: post
title: "코드 강조하기"
description: "code highlighting features"
categories: [DevEnviroment]
tags: [gitblog]
redirect_from:
  - /2020/07/13/
---

* Kramdown table of contents
{:toc .toc}

# Fenced Code Blocks

~~~~~~~~~~~~
~~~~~~~
code with tildes
~~~~~~~~
~~~~~~~~~~~~~~~~~~

# Simple codeblock

    function myFunction() {
        alert("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.");
    }


# Highlighted
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


## External Gist

<script src="https://gist.github.com/yizeng/9b871ad619e6dcdcc0545cac3101f361.js"></script>
