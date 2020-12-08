import pathlib

import jinja2

env = jinja2.Environment(
    block_start_string="<%",
    block_end_string="%>",
    loader=jinja2.PackageLoader("generator"),
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)

template = env.get_template("day.md")

solutions_path = pathlib.Path("./solutions")

contents = {}

for solution_dir in sorted(solutions_path.iterdir()):
    day = int(solution_dir.stem)
    python_files = [file for file in solution_dir.glob("*.py")]
    contents[day] = {}
    for i, python_file in enumerate(python_files):
        content = open(python_file, "r").read()
        contents[day][python_file.name] = content

output = template.render(contents=contents)
with open("index.md", "w") as file:
    file.write(output)
