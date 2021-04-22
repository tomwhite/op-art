# Linear Algebra Functions
# https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html#

# Good animations are hard to implement for these functions!

from itertools import product
import numpy as np
from numpy.compat import basestring
#from numpy.core.einsumfunc import _parse_einsum_input

from ._array_object import Array, ArrayRepresentation, CellRepresentation


einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
einsum_symbols_set = set(einsum_symbols)

# TODO improve
def asarray(o):
    return o

# We need to copy this from numpy to ensure asarray works with our Array
# Perhaps we can fix this?

# This function duplicates numpy's _parse_einsum_input() function
# See https://github.com/numpy/numpy/blob/master/LICENSE.txt
# or NUMPY_LICENSE.txt within this directory
def _parse_einsum_input(operands):
    """
    A reproduction of numpy's _parse_einsum_input()
    which in itself is a reproduction of
    c side einsum parsing in python.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction
    Examples
    --------
    The operand list is simplified to reduce printing:
    >> a = np.random.rand(4, 4)
    >> b = np.random.rand(4, 4, 4)
    >> __parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b])
    >> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b])
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], basestring):
        subscripts = operands[0].replace(" ", "")
        operands = [asarray(o) for o in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in einsum_symbols_set:
                raise ValueError("Character %s is not a valid symbol." % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError(
                        "For this input type lists must contain "
                        "either int or Ellipsis"
                    )
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError(
                        "For this input type lists must contain "
                        "either int or Ellipsis"
                    )
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(sub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace("...", rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in einsum_symbols_set:
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols_set:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input" % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError(
            "Number of einsum subscripts must be equal to the number of operands."
        )

    return (input_subscripts, output_subscript, operands)

def einsum(*operands, **kwargs):

    def extract_array(o):
        return o.arr if isinstance(o, Array) else o

    arr = np.einsum(*[extract_array(o) for o in operands], **kwargs)

    inputs, outputs, ops = _parse_einsum_input(operands)

    inputs = [tuple(i) for i in inputs.split(",")]
    outputs = tuple(outputs)

    all_inds = set(a for i in inputs for a in i)
    contract_inds = all_inds - set(outputs)

    # Don't need to check dimensions are consistent since underlying implementation does
    dimension_dict = {}
    for operand, input in zip(ops, inputs):
        for cnum, char in enumerate(input):
            dimension_dict[char] = operand.shape[cnum]

    def sub(input, d):
        # substitute index labels in the input with values from the dictonary
        return tuple(d.get(i, i) for i in input)

    def get_sources(output_ind):
        # substitute fixed output indices
        fixed_d = {ind: val for ind, val in zip(outputs, output_ind)}
        inputs_sub = [sub(input, fixed_d) for input in inputs]
        # substitute contraction indices
        for con in product(*[range(dimension_dict[ind]) for ind in contract_inds]):
            contract_d = {ind: val for ind, val in zip(contract_inds, con)}
            yield [sub(input, contract_d) for input in inputs_sub]

    new_rep = Array._get_representation(arr)
    cells = []
    for cell in new_rep.cells:
        sources = list(get_sources(cell.index))
        sources_by_input = list(zip(*sources))
        all_sources = []
        for operand, input_indexes in zip(ops, sources_by_input):
            cell_ids = [cell.id for cell in operand.representation.cells if cell.index in input_indexes]
            all_sources.extend(cell_ids)
        
        cell = CellRepresentation(cell.id, cell.index, cell.value, all_sources)
        cells.append(cell)

    rep = ArrayRepresentation(new_rep.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))
    return Array.from_representation(arr, rep)


def matmul(x1, x2, /):
    if x1.ndim != 2 or x2.ndim != 2:
        raise NotImplementedError("matmul only implemented for 2x2 arrays")

    return einsum("...ij,...jk->...ik", x1, x2)

def tensordot(x1, x2, /, *, axes=2):
    if isinstance(axes, int):
        axes = (range(-1, -1 - axes, -1), range(axes))
    # inspired by https://scicomp.stackexchange.com/a/34720
    x1_indexes = list(range(x1.ndim))
    x2_indexes = list(range(x1.ndim, x1.ndim + x2.ndim))
    for x1_ind, x2_ind in zip(*axes):
        x2_indexes[x2_ind] = x1_indexes[x1_ind]
    return einsum(x1, x1_indexes, x2, x2_indexes)

def transpose(x, /, *, axes=None):
    x_indexes = list(range(x.ndim))
    if axes is None:
        output_indexes = list(reversed(range(x.ndim)))
    else:
        output_indexes = list(axes)
    return einsum(x, x_indexes, output_indexes)