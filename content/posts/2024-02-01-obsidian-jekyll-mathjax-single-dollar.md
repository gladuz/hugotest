---
title: "How to make Obsidian and Jekyll equations compatible"
date: "2024-02-01"
description: "Fixing the single dollar display compatibility between Obsidian and Jekyll"
tags:
  - obsidian
  - jekyll
---


# TL;DR
Set `processEscapes` to `False` or remove it from `tax` dictionary
```html
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
```

# What was not working
I've tried to write blog posts using Obsidian for my Jekyll website. The issue was the Mathjax version I was using didn't recognize single dollar sign for inline equations.

I was using this Mathjax setup taken from the documentation:

```html
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    processEscapes: true
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
```
But the \\$ signs in my markdown was displaying as \\$ in the rendered HTML. Some of the related answers mentioned the issue of markdown + Mathjax processing order. 

I've tried different `inlineMath` options, but at the end downgrading to `2.x` version solved the problem. 
```html
 <script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js?config=TeX-MML-AM_CHTML">
 </script>
 <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      extensions: ["tex2jax.js"],
      jax: ["input/TeX", "output/HTML-CSS"],
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      },
      "HTML-CSS": { availableFonts: ["TeX"] }
    });
 </script>
```

However after trying with options later, turns out just removing `processEscapes` seem to solve the problem.