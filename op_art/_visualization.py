# Functions for visualizing array operations using HTML and JavaScript

import inspect
import json
import os
from dataclasses import asdict, dataclass
from textwrap import dedent
from typing import Any, Tuple

import numpy as np
import numpy.array_api as nxp
from array_tracker._array_object import get_arrays


@dataclass()
class CellRepresentation:
    id: Any
    index: Any
    value: Any
    sources: Any = None


@dataclass()
class ArrayRepresentation:
    id: int
    kind: str
    ndim: int
    shape: Any
    cells: Tuple[CellRepresentation]


def _get_representation(array):
    """Convert array into a representation suitable for visualization"""
    arr = array.arr
    cells = []
    it = np.nditer(arr, flags=["multi_index", "refs_ok", "zerosize_ok"], order="C")
    for x in it:
        ind = it.multi_index
        offset = array.offsets[ind]
        cell_id = f"{array.id}_{offset}"
        if array.src_arr_ids is None:
            cell_sources = None
        else:
            if array.offsets.ndim == array.src_arr_ids.ndim:
                src_ind = ind
            else:
                src_ind = ind + (Ellipsis,)
            src_arr_id = array.src_arr_ids[src_ind]
            if src_arr_id.ndim == 0:
                if src_arr_id == -1:
                    cell_sources = None
                else:
                    src_offset = array.src_offsets[src_ind]
                    cell_sources = [f"{src_arr_id}_{src_offset}"]
            else:
                if nxp.all(src_arr_id == -1):
                    cell_sources = None
                else:
                    src_offset = array.src_offsets[src_ind]
                    cell_sources = [
                        f"{i}_{o}" for i, o in zip(src_arr_id, src_offset) if i != -1
                    ]
        cells.append(
            CellRepresentation(cell_id, it.multi_index, x.item(), cell_sources)
        )
    return ArrayRepresentation(
        array.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells)
    )


def get_array_id_to_representation():
    # TODO: consider tracking graph of arrays to get connected part to cut down number processed in rewrite_sources
    return {id: _get_representation(array) for id, array in get_arrays().items()}


def _rewrite_sources(source_ids, visible_ids, cell_id_to_sources):
    if source_ids is None:
        return []
    res = []
    for si in source_ids:
        arr_id = int(si.split("_")[0])
        if arr_id in visible_ids:
            res.append(si)
        else:
            res.extend(
                _rewrite_sources(
                    cell_id_to_sources[si], visible_ids, cell_id_to_sources
                )
            )
    return res


def rewrite_sources(source_ids, visible_ids, cell_id_to_sources):
    """Rewrite sources so none are from invisible arrays"""
    x = _rewrite_sources(source_ids, visible_ids, cell_id_to_sources)
    return x if len(x) > 0 else None


def rewrite_representation(arr_rep, visible_ids=None, array_id_to_representation=None):
    """Rewrite representation so it doesn't include any invisible arrays"""
    if visible_ids is None:
        return arr_rep
    if array_id_to_representation is None:
        array_id_to_representation = get_array_id_to_representation()
    cell_id_to_sources = {}
    for representation in array_id_to_representation.values():
        cells = representation.cells
        for cell in cells:
            cell_id_to_sources[cell.id] = cell.sources
    new_cells = []
    for cell in arr_rep.cells:
        new_sources = rewrite_sources(cell.sources, visible_ids, cell_id_to_sources)
        new_cell = CellRepresentation(cell.id, cell.index, cell.value, new_sources)
        new_cells.append(new_cell)
    return ArrayRepresentation(
        arr_rep.id, arr_rep.kind, arr_rep.ndim, arr_rep.shape, new_cells
    )


def array_to_json_dict(array, visible_ids=None, array_id_to_representation=None):
    if array_id_to_representation is None:
        array_id_to_representation = get_array_id_to_representation()
    return asdict(
        rewrite_representation(
            array_id_to_representation[array.id],
            visible_ids,
            array_id_to_representation,
        )
    )


def arrays_to_json(arrays):
    visible_ids = [a.id for a in arrays]
    array_id_to_representation = get_array_id_to_representation()
    return json.dumps(
        [
            array_to_json_dict(a, visible_ids, array_id_to_representation)
            for a in arrays
        ],
        indent=2,
    )


def strings_to_json(strings):
    return json.dumps(strings, indent=2)


def boolean_to_js(v):
    return "true" if v else "false"


def arrays_to_html(
    arrays, lines, base_url=".", animate=True, rankdir="TB", show_values=True
):
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
</html>""" % (
        require_js_path,
        op_art_css_path,
        base_url,
        arrays_to_json(arrays),
        strings_to_json(lines),
        boolean_to_js(animate),
        rankdir,
        boolean_to_js(show_values),
    )


def write_html(
    filename, arrs, lines, base_url=".", animate=True, rankdir="TB", show_values=True
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(
            arrays_to_html(
                arrs,
                lines,
                base_url=base_url,
                animate=animate,
                rankdir=rankdir,
                show_values=show_values,
            )
        )


def get_source(fn):
    """Get the source code for a function (excluding declaration and return line."""
    src = inspect.getsource(fn)
    src = src.strip()
    lines = src.split("\n")
    lines = lines[1:-1]  # remove function declaration and return
    src = dedent("\n".join(lines))
    return src.split("\n")


def visualize(*arrays, animate=True, rankdir="TB", show_values=True):
    import inspect

    from IPython.display import Javascript, display

    frame = inspect.currentframe()
    frame = frame.f_back  # go back one in the stack
    src = inspect.getsource(frame)
    src = src.strip()
    last_line = src.split("\n")[-1].strip()
    # TODO: improve
    vars = last_line[last_line.index("(") + 1 :]
    vars = vars[: vars.index(")")]
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
