# Advent of Code 2020
<% for day, day_contents in contents.items() %>
<details><summary>View</summary>
## Day {{day}}
<% for part, content in day_contents.items() %>
### {{part}}
<details><summary>Code</summary>
{% highlight python %}
{{ content }}
{% endhighlight %}
</details>
<% endfor %>
</details>
<% endfor %>
