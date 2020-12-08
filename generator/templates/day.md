# Advent of Code 2020
<% for day, day_contents in contents.items() %>
# Day {{day}}

## Solutions
<% for part, content in day_contents.items() %>
### {{part}}

<details><summary>Code</summary>
{% highlight python %}
{{ content }}
{% endhighlight %}
</details>

<% endfor %>
<% endfor %>
