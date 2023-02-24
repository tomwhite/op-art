# flake8: noqa

try:
    from ._jupyter import init_jupyter

    init_jupyter()
except ImportError:
    pass

from array_tracker import *

from ._visualization import arrays_to_html, get_source, visualize, write_html
