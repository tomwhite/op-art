import shutil

import op_art
import op_art as xp

# Array object

def T_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = a.T
    return a, b

array_object = [T_example]

# Indexing

def dimensions_example():
    a = xp.asarray(0)
    b = xp.reshape(xp.arange(2), (2,))
    c = xp.reshape(xp.arange(6), (3, 2))
    d = xp.reshape(xp.arange(24), (4, 3, 2))
    return a, b, c, d

def single_axis_example():
    a = xp.arange(6)
    b = a[1:3]
    return a, b

def multi_axis_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = a[:, 1]
    return a, b

def boolean_array_example():
    a = xp.arange(6)
    b = xp.asarray([True, False, True, False, True, False])
    c = a[b]
    return a, b, c

indexing = [dimensions_example, single_axis_example, multi_axis_example, boolean_array_example]

# Data type functions

def broadcast_arrays_example():
    a = xp.asarray([[1, 2, 3]])
    b = xp.asarray([[4], [5]])
    c, d = xp.broadcast_arrays(a, b)
    return a, b, c, d

def broadcast_to_example():
    a = xp.ones((1,))
    b = xp.broadcast_to(a, (1, 2))
    return a, b

data_type_functions = [broadcast_arrays_example, broadcast_to_example]

# Creation Functions

def arange_example():
    a = xp.arange(6)
    return (a,)

def asarray_example():
    a = xp.asarray([5, 1, 0, 3, 2, 4])
    return (a,)

def empty_example():
    a = xp.empty((1, 2))
    return (a,)

def empty_like_example():
    a = xp.asarray([5, 2, 4, 1])
    b = xp.empty_like(a)
    return a, b

def eye_example():
    a = xp.eye(3, 2)
    return (a,)

def full_example():
    a = xp.full((1, 2), 2)
    return (a,)

def full_like_example():
    a = xp.asarray([5, 2, 4, 1])
    b = xp.full_like(a, 2)
    return a, b

def linspace_example():
    a = xp.linspace(2.0, 3.0, num=5)
    return (a,)

def ones_example():
    a = xp.ones((1, 2))
    return (a,)

def ones_like_example():
    a = xp.asarray([5, 2, 4, 1])
    b = xp.ones_like(a)
    return a, b

def zeros_example():
    a = xp.zeros((1, 2))
    return (a,)

def zeros_like_example():
    a = xp.asarray([5, 2, 4, 1])
    b = xp.zeros_like(a)
    return a, b

creation_functions = [arange_example, asarray_example, empty_example, empty_like_example, eye_example, full_example, full_like_example, linspace_example, ones_example, ones_like_example, zeros_example, zeros_like_example]

# Manipulation Functions

def concat_example():
    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[5], [6]])
    c = xp.concat((a, b), axis=1)
    return a, b, c

def expand_dims_example():
    a = xp.arange(6)
    b = xp.expand_dims(a, axis=1)
    return a, b

def flip_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.flip(a, axis=0)
    return a, b

def reshape_example():
    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))
    return a, b

def roll_example():
    a = xp.arange(6)
    b = xp.roll(a, 2)
    return a, b

def squeeze_example():
    a = xp.asarray([[0], [1], [1], [1], [0], [1]])
    b = xp.squeeze(a, axis=1)
    return a, b

def stack_example():
    a = xp.asarray([1, 2, 3])
    b = xp.asarray([2, 3, 4])
    c = xp.stack((a, b))
    return a, b, c

manipulation_functions = [concat_example, expand_dims_example, flip_example, reshape_example, roll_example, squeeze_example, stack_example]

# Element-wise Functions

def add_example():
    a = xp.ones((1, 2))
    b = xp.ones((1, 2))
    c = xp.add(a, b)
    return a, b, c

def add_broadcast_example():
    a = xp.asarray([0, 1, 2, 3, 4])
    b = xp.ones((1,))
    c = xp.add(a, b)
    return a, b, c

def negative_example():
    a = xp.arange(6)
    b = xp.negative(a)
    return a, b

elementwise_functions = [add_example, add_broadcast_example, negative_example]

# Statistical Functions

def max_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.max(a, axis=0)
    return a, b

def mean_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.mean(a, axis=0)
    return a, b

def min_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.min(a, axis=0)
    return a, b

def prod_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.prod(a, axis=0)
    return a, b

def std_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.std(a, axis=0)
    return a, b

def sum_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.sum(a, axis=0)
    return a, b

def var_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.var(a, axis=0)
    return a, b

statistical_functions = [max_example, mean_example, min_example, prod_example, std_example, sum_example, var_example]

# Linear Algebra Functions

def einsum_example():
    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[1, 2], [3, 4]])
    c = xp.einsum("ij,jk->ik", a, b)
    return a, b, c

def matmul_example():
    a = xp.asarray([[0, 1, 2], [3, 4, 5]])
    b = xp.asarray([[5, 1], [0, 3], [2, 4]])
    c = xp.matmul(a, b)
    return a, b, c

def tensordot_example():
    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[1, 2], [3, 4]])
    c = xp.tensordot(a, b)
    return a, b, c

def transpose_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.transpose(a)
    return a, b

linear_algebra_functions = [einsum_example, matmul_example, tensordot_example, transpose_example]

# Searching Functions

def argmax_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.argmax(a, axis=0)
    return a, b

def argmin_example():
    a = xp.reshape(xp.arange(6), (3, 2))
    b = xp.argmin(a, axis=0)
    return a, b

searching_functions = [argmax_example, argmin_example]

# Sorting Functions

def argsort_example():
    a = xp.asarray([5, 1, 0, 3, 2, 4])
    b = xp.argsort(a)
    return a, b

def sort_example():
    a = xp.asarray([5, 1, 0, 3, 2, 4])
    b = xp.sort(a)
    return a, b

sorting_functions = [argsort_example, sort_example]

# Set Functions

def unique_example():
    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b = xp.unique(a)
    return a, b

set_functions = [unique_example]

# Utility Functions

def all_example():
    a = xp.reshape(xp.asarray([0, 1, 1, 1, 0, 1]), (3, 2))
    b = xp.all(a, axis=0)
    return a, b

def any_example():
    a = xp.reshape(xp.asarray([0, 1, 1, 1, 0, 1]), (3, 2))
    b = xp.any(a, axis=0)
    return a, b

utility_functions = [all_example, any_example]

# Misc

def reshape_flip_reshape():
    a = xp.arange(24)
    b = xp.reshape(a, (3, 4, 2))
    c = xp.flip(b, axis=0)
    d = xp.reshape(c, 24)
    return a, b, c, d

misc_examples = [reshape_flip_reshape]

all_examples = {
    "array_object": array_object,
    "indexing": indexing,
    "data_type_functions": data_type_functions,
    "creation_functions": creation_functions,
    "manipulation_functions": manipulation_functions,
    "elementwise_functions": elementwise_functions,
    "statistical_functions": statistical_functions,
    "linear_algebra_functions": linear_algebra_functions,
    "searching_functions": searching_functions,
    "sorting_functions": sorting_functions,
    "set_functions": set_functions,
    "utility_functions": utility_functions,
    "misc": misc_examples
}

if __name__ == "__main__":

    # copy css and js to docs directory, so it is self-contained
    shutil.copy("op_art/web/require.js", "docs/require.js")
    shutil.copy("op_art/web/op_art.css", "docs/op_art.css")
    shutil.copy("op_art/web/op_art.js", "docs/op_art.js")

    for category, functions in all_examples.items():
        for example in functions:
            example_name = example.__name__.replace("_example", "")

            op_art.reset_ids()
            vars = example()

            lines = op_art.get_source(example)
            op_art.write_html(f"docs/{category}/{example_name}.html", vars, lines, base_url="..")
