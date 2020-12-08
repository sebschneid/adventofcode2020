# Advent of Code 2020

<% for day, day_contents in contents.items() %>

## Day {{day}}

<% for file, content in day_contents.items() %>

### {{file}}

<details><summary>Code</summary>

```python
{{ content }}
```

</details>
</br>

<% endfor %>

</details>

</br>
<hr>
</br>
<% endfor %>
