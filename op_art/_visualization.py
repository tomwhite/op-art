# Functions for visualizing array operations using HTML and JavaScript

import inspect
import json
import os
from textwrap import dedent

from . import _array_object
from ._array_object import ArrayRepresentation, CellRepresentation

def _rewrite_sources(source_ids, visible_ids, cell_id_to_sources):
    if source_ids is None:
        return []
    res = []
    for si in source_ids:
        arr_id = int(si.split("_")[0])
        if arr_id in visible_ids:
            res.append(si)
        else:
            res.extend(_rewrite_sources(cell_id_to_sources[si], visible_ids, cell_id_to_sources))
    return res

def rewrite_sources(source_ids, visible_ids):
    """Rewrite sources so none are from invisible arrays"""
    cell_id_to_sources = {}
    for array_id, array in _array_object.id_to_array.items():
        cells = array.representation.cells
        for cell in cells:
            cell_id_to_sources[cell.id] = cell.sources

    x = _rewrite_sources(source_ids, visible_ids, cell_id_to_sources)
    return x if len(x) > 0 else None

def rewrite_representation(arr_rep, visible_ids=None):
    """Rewrite representation so it doesn't include any invisible arrays"""
    if visible_ids is None:
        return arr_rep
    new_cells = []
    for cell in arr_rep.cells:
        new_sources = rewrite_sources(cell.sources, visible_ids)
        new_cell = CellRepresentation(cell.id, cell.index, cell.value, new_sources)
        new_cells.append(new_cell)
    return ArrayRepresentation(arr_rep.id, arr_rep.kind, arr_rep.ndim, arr_rep.shape, new_cells)

def arrays_to_json(arrays):
    visible_ids = [a.id for a in arrays]
    return json.dumps([a.to_json(visible_ids) for a in arrays], indent=2)

def strings_to_json(strings):
    return json.dumps(strings, indent=2)

def boolean_to_js(v):
    return "true" if v else "false"

def arrays_to_html(arrays, lines, base_url=".", animate=True, rankdir="TB", show_values=True):
    require_js_path = f"{base_url}/require.js"
    op_art_css_path = f"{base_url}/op_art.css"
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>op-art</title>
    <script src="%s"></script>
    <link href="%s" rel="stylesheet" />
  </head>

  <body>
    <div id="plot"></div>

    <script>
      require.config({
        baseUrl: "%s",
        paths: {
          d3: 'https://d3js.org/d3.v5.min'
        }
      });
      require(["op_art"], function(op_art) {
        const arrs = %s;
        const lines = %s;

        op_art.visualize("#plot", arrs, lines, 1500, %s, "%s", %s);
      });
    </script>
  </body>
</html>""" % (require_js_path, op_art_css_path, base_url, arrays_to_json(arrays), strings_to_json(lines), boolean_to_js(animate), rankdir, boolean_to_js(show_values))

def write_html(filename, arrs, lines, base_url=".", animate=True, rankdir="TB", show_values=True):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(arrays_to_html(arrs, lines, base_url=base_url, animate=animate, rankdir=rankdir, show_values=show_values))


def get_source(fn):
    """Get the source code for a function (excluding declaration and return line."""
    src = inspect.getsource(fn)
    src = src.strip()
    lines = src.split("\n")
    lines = lines[1:-1]  # remove function declaration and return
    src = dedent("\n".join(lines))
    return src.split("\n")


def visualize(*arrays, animate=True, rankdir="TB", show_values=True):
    from IPython.display import display, Javascript
    import inspect
    frame = inspect.currentframe()
    frame = frame.f_back # go back one in the stack
    src = inspect.getsource(frame)
    src = src.strip()
    last_line = src.split("\n")[-1].strip()
    # TODO: improve
    vars = last_line[last_line.index("(") + 1:]
    vars = vars[:vars.index(")")]
    vars = [v.strip() for v in vars.split(",")]

    js = """(function(element){
    require(['op_art'], function(op_art) {
        op_art.visualize(element.get(0), %s, %s, 1500, %s, "%s", %s);
    });
})(element);""" % (
        arrays_to_json(arrays),
        vars,
        boolean_to_js(animate),
        rankdir,
        boolean_to_js(show_values),
    )

    return display(Javascript(js))