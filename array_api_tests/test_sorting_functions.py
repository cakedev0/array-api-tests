import cmath
from collections import defaultdict
from typing import List, Set

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.control import assume

from . import _array_module as xp
from . import dtype_helpers as dh
from . import hypothesis_helpers as hh
from . import pytest_helpers as ph
from . import shape_helpers as sh
from .typing import Scalar, Shape


def assert_sets(
    func_name: str,
    indices: List[Shape],
    out: Set[Scalar],
    expected: Set[Scalar],
    kw={},
):
    out_repr = f"out[{indices}]"
    if cmath.isnan(out):
        raise NotImplementedError()
    diff = out.difference(expected)
    msg = f"{out_repr}={out}, but should be in {expected} (diff={diff}) [{func_name}({ph.fmt_kw(kw)})]"
    assert out == expected, msg


# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@pytest.mark.unvectorized
@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
        shape=hh.shapes(min_dims=1, min_side=1, max_side=50),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_argsort(x, data):
    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    kw = data.draw(
        hh.kwargs(
            axis=st.integers(-x.ndim, x.ndim - 1),
            descending=st.booleans(),
            stable=st.booleans(),
        ),
        label="kw",
    )

    out = xp.argsort(x, **kw)

    ph.assert_default_index("argsort", out.dtype)
    ph.assert_shape("argsort", out_shape=out.shape, expected=x.shape, kw=kw)
    axis = kw.get("axis", -1)
    axes = sh.normalize_axis(axis, x.ndim)
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices in sh.axes_ndindex(x.shape, axes):
        elements_to_sort = [scalar_type(x[idx]) for idx in indices]
        sorted_indices = [int(out[idx]) for idx in indices]
        expected_indices = list(range(len(elements_to_sort)))
        expected_indices.sort(
            key=elements_to_sort.__getitem__, reverse=kw.get("descending", False)
        )
        if kw.get("stable", True):
            for x_idx, actual, expected in zip(indices, sorted_indices, expected_indices):
                ph.assert_scalar_equals(
                    "argsort", type_=int, idx=x_idx, kw=kw,
                    out=actual, expected=expected
                )
            continue

        expected_sets = defaultdict(set)
        for i in expected_indices:
            e = elements_to_sort[i]
            expected_sets[e].add(i)

        actual_sets = defaultdict(set)
        indices_in_out = defaultdict(list)
        for i, idx in zip(sorted_indices, indices):
            e = elements_to_sort[i]
            actual_sets[e].add(i)
            indices_in_out[e].append(idx)

        for e in set(elements_to_sort):
            assert expected_sets[e] == actual_sets[e]


@pytest.mark.unvectorized
# TODO: Test with signed zeros and NaNs (and ignore them somehow)
@given(
    x=hh.arrays(
        dtype=hh.real_dtypes,
        shape=hh.shapes(min_dims=1, min_side=1),
        elements={"allow_nan": False},
    ),
    data=st.data(),
)
def test_sort(x, data):
    if dh.is_float_dtype(x.dtype):
        assume(not xp.any(x == -0.0) and not xp.any(x == +0.0))

    kw = data.draw(
        hh.kwargs(
            axis=st.integers(-x.ndim, x.ndim - 1),
            descending=st.booleans(),
            stable=st.booleans(),
        ),
        label="kw",
    )

    out = xp.sort(x, **kw)

    ph.assert_dtype("sort", out_dtype=out.dtype, in_dtype=x.dtype)
    ph.assert_shape("sort", out_shape=out.shape, expected=x.shape, kw=kw)
    axis = kw.get("axis", -1)
    axes = sh.normalize_axis(axis, x.ndim)
    scalar_type = dh.get_scalar_type(x.dtype)
    for indices in sh.axes_ndindex(x.shape, axes):
        elements = [scalar_type(x[idx]) for idx in indices]
        size = len(elements)
        orders = sorted(
            range(size), key=elements.__getitem__, reverse=kw.get("descending", False)
        )
        for out_idx, o in zip(indices, orders):
            x_idx = indices[o]
            # TODO: error message when unstable should not imply just one idx
            ph.assert_0d_equals(
                "sort",
                x_repr=f"x[{x_idx}]",
                x_val=x[x_idx],
                out_repr=f"out[{out_idx}]",
                out_val=out[out_idx],
                kw=kw,
            )
