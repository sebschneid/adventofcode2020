# Advent of Code 2020

<% for day, day_contents in contents.items() %>

## Day {{day}}

<details><summary>View</summary>

<% for part, content in day_contents.items() %>

### {{part}}

<details><summary>Code</summary>

```python
{{ content }}
```

</details>
<% endfor %>
</details>
<% endfor %>
