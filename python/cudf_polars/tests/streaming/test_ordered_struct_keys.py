# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise", raise_on_fail=True)
    )


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_multi_column_ordered_window(engine, mixed, empty):
    # Multi-column order_by becomes as_struct in the expression IR. Test the
    # Q47/Q57 combination of windows with different grouping keys as well as
    # the ordered rank alone, preserving the original input row alignment.
    q = pl.LazyFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "year": [2000, 2000, 2001, 2000, 2001, 2001],
            "month": [2, 1, 1, 1, 2, 1],
            "x": [1.0, None, 3.0, 4.0, 5.0, 6.0],
        }
    )
    if empty:
        q = q.clear()
    expressions = [
        pl.col("year")
        .rank(method="ordinal")
        .over("g", order_by=["year", "month"])
        .alias("rn")
    ]
    if mixed:
        expressions.append(pl.col("x").mean().over(["g", "year"]).alias("avg"))
    assert_gpu_result_equal(q.with_columns(expressions), engine=engine)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_ordered_window_after_replicated_top_k(engine, empty, mixed):
    # A distributed top-k emits a replicated result. Ordered windows must
    # preserve that row ownership so sibling mean/passthrough columns align.
    q = pl.LazyFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b", "c"],
            "year": [2000, 2000, 2001, 2000, 2001, 2001, 2000],
            "month": [2, 1, 1, 1, 2, 1, 1],
            "x": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    if empty:
        q = q.clear()
    # Keep g out of the leading ordering keys so the ordered g window
    # exercises its forward/return shuffle rather than the chunkwise path.
    q = q.sort(["year", "month", "g"]).head(6)
    expressions = [
        pl.col("year")
        .rank(method="ordinal")
        .over("g", order_by=["year", "month"])
        .alias("rn")
    ]
    if mixed:
        expressions.append(pl.col("x").mean().over(["g", "year"]).alias("avg"))
    assert_gpu_result_equal(
        q.with_columns(expressions), engine=engine, check_row_order=True
    )
