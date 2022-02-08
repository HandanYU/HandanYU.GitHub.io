---
layout: post
title: How to build blog using jekyll
feature-img: 
summary: The tutorial to tell you how to build blog using jekll
featured-img: jekyll
language: english 
category: others
---

1. markdown latex

首先在 /_layouts/post.html中的<head>...</head>之间添加如下代码

```html
  <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$']],
        displayMath: [['$$', '$$']],
        processEnvironments: true,
        processRefs: true
      },
      options: {
        skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
        ignoreHtmlClass: 'tex2jax_ignore',
        renderActions: {
          find_script_mathtex: [10, function (doc) {
            for (const node of document.querySelectorAll('script[type^="math/tex"]')) {
              const display = !!node.type.match(/; *mode=display/);
              const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
              const text = document.createTextNode('');
              node.parentNode.replaceChild(text, node);
              math.start = { node: text, delim: '', n: 0 };
              math.end = { node: text, delim: '', n: 0 };
              doc.math.push(math);
            }
          }, '']
        }
      },
      svg: {
        fontCache: 'global'
      }
    };
  </script>
```

- 
行外公式

```
❌

test test test

$$
formula
$$

test test

```

```
🙆

test test test

$$formula$$

test test

```

行内公式

```
test test test $$formula$$ test test test
```


对于|，需要单独使用\|，且不能包含在$$..$$之间，否则会呈现||

## 停止jeklly serve后无法再次启动的原因

```
jekyll new <blog name>

cd ./<blog name>

jekyll serve
```


将文件夹中除了_site, gem相关的文件，其余文件复制到新创建的jekyll blog