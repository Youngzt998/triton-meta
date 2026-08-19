"""One evaluation step per file.

`artifact.py` imports every `steps/<x>.py` whose name does not start with `_`, so a step is
added, implemented or removed by touching exactly one file. Read `_common.py` for the contract a
step module has to meet and for the helpers a step is expected to reuse rather than re-implement.
"""
