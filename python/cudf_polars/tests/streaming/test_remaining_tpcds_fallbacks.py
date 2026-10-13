# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2, fallback_mode="raise", raise_on_fail=True
        )
    )


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_complete_frame_rank(engine, descending, empty):
    q = pl.LazyFrame({"id": range(7), "x": [2.0, None, 1.0, 2.0, 5.0, 1.0, 3.0]})
    if empty:
        q = q.clear()
    # Top-k produces a complete replicated input; rankings retain ties/nulls.
    q = (
        q.sort("id")
        .head(7)
        .with_columns(pl.col("x").rank(method="min", descending=descending).alias("r"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.engine_params(["ray"])
def test_distributed_incomplete_frame_rank_still_raises(engine):
    q = pl.LazyFrame({"id": range(7), "x": [2.0, None, 1.0, 2.0, 5.0, 1.0, 3.0]})
    q = q.with_columns(pl.col("x").rank(method="min").alias("r"))
    # The runtime guard must not turn an unsupported distributed rank into an
    # unreported gather. Match the actual ownership check, not just any error.
    with pytest.raises(Exception) as error:
        assert_gpu_result_equal(q, engine=engine)
    assert "not already complete on every rank" in str(error.value) + repr(error.value)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_scalar_conditional_join(engine, reverse, empty):
    left = pl.LazyFrame({"id": range(7), "x": [2.0, None, 1.0, 2.0, 5.0, 1.0, 3.0]})
    if empty:
        left = left.clear()
    right = pl.LazyFrame({"z": [1.0, 3.0, None]}).select(
        pl.col("z").mean().alias("threshold")
    )
    if reverse:
        left, right = right, left
    q = left.join_where(right, pl.col("x") > pl.col("threshold"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("empty_side", [None, "left", "right"])
def test_cross_join_schema_and_empty_inputs(engine, empty_side):
    left = pl.LazyFrame({"id": [1, 2, 3], "__0": [10, 20, 30], "v": [None, 2, 3]})
    right = pl.LazyFrame({"id": [4, 5], "__1_suffix": [40, 50]})
    if empty_side == "left":
        left = left.clear()
    if empty_side == "right":
        right = right.clear()
    q = left.join(right, how="cross", suffix="_suffix")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_q9_masked_pivot_preserves_missing_and_null_buckets(engine, monkeypatch):
    from cudf_polars.streaming.benchmarks.pdsds_queries import q9

    data = {
        "store_sales": pl.LazyFrame(
            {
                "ss_quantity": [1, 5, 45, 101, None],
                "ss_net_profit": [None, None, 5.0, 1.0, 1.0],
                "ss_ext_sales_price": [1.0, 2.0, 10.0, 5.0, 2.0],
            }
        ),
        "reason": pl.LazyFrame({"r_reason_sk": [1, 2]}),
    }
    monkeypatch.setattr(q9, "get_data", lambda _, table, suffix: data[table])
    monkeypatch.setattr(
        q9,
        "load_parameters",
        lambda *args, **kwargs: {
            "aggcthen": "ss_net_profit",
            "aggcelse": "ss_ext_sales_price",
            "rc": [0, 0, 5, 0, 0],
        },
    )
    config = SimpleNamespace(
        scale_factor=1, qualification=False, dataset_path="", suffix=""
    )
    q = q9.polars_impl(config).frame
    expected = pl.DataFrame(
        {
            "bucket1": [None],
            "bucket2": [None],
            "bucket3": [10.0],
            "bucket4": [None],
            "bucket5": [None],
        },
        schema={f"bucket{i}": pl.Float64 for i in range(1, 6)},
    )
    assert_frame_equal(q.collect(), expected)
    assert_gpu_result_equal(q, engine=engine)


def test_q16_exists_rewrite_preserves_duplicates_and_nulls(engine, monkeypatch):
    from datetime import date

    from cudf_polars.streaming.benchmarks.pdsds_queries import q16

    sales = pl.LazyFrame(
        {
            "cs_order_number": [
                10,
                10,
                10,
                11,
                11,
                12,
                12,
                13,
                13,
                14,
                14,
                15,
                15,
                16,
                20,
                20,
                None,
                None,
            ],
            "cs_warehouse_sk": [
                1,
                1,
                2,
                1,
                None,
                None,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                None,
                1,
                2,
                1,
                2,
            ],
            "cs_ship_date_sk": [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "cs_ship_addr_sk": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            "cs_call_center_sk": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
            "cs_ext_ship_cost": [
                5.0,
                7.0,
                99.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                6.0,
                99.0,
                4.0,
                99.0,
                1.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ],
            "cs_net_profit": [
                2.0,
                3.0,
                99.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                1.0,
                99.0,
                2.0,
                99.0,
                1.0,
                99.0,
                99.0,
                99.0,
                99.0,
            ],
        }
    )
    dates = pl.LazyFrame(
        {"d_date_sk": [1, 2], "d_date": [date(2000, 1, 1), date(2000, 6, 1)]}
    )
    addresses = pl.LazyFrame({"ca_address_sk": [1, 2], "ca_state": ["CA", "NY"]})
    centers = pl.LazyFrame({"cc_call_center_sk": [1, 2], "cc_county": ["X", "Y"]})
    returns = pl.LazyFrame({"cr_order_number": [20, None]})
    data = {
        "catalog_sales": sales,
        "catalog_returns": returns,
        "date_dim": dates,
        "customer_address": addresses,
        "call_center": centers,
    }
    monkeypatch.setattr(q16, "get_data", lambda _, table, suffix: data[table])
    monkeypatch.setattr(
        q16,
        "load_parameters",
        lambda *args, **kwargs: {
            "year": 2000,
            "month": 1,
            "state": "CA",
            "county": ["X"],
        },
    )
    config = SimpleNamespace(
        scale_factor=1, qualification=False, dataset_path="", suffix=""
    )
    q = q16.polars_impl(config).frame
    # Original row-index/self-join semantics, including an other warehouse
    # outside the date/address/call-center filters and null join keys.
    filtered = (
        sales.join(dates, left_on="cs_ship_date_sk", right_on="d_date_sk")
        .join(addresses, left_on="cs_ship_addr_sk", right_on="ca_address_sk")
        .join(centers, left_on="cs_call_center_sk", right_on="cc_call_center_sk")
        .filter(
            (pl.col("d_date") < date(2000, 3, 1))
            & (pl.col("ca_state") == "CA")
            & (pl.col("cc_county") == "X")
        )
    )
    exists = (
        filtered.with_row_index("row_id")
        .join(
            sales.select("cs_order_number", pl.col("cs_warehouse_sk").alias("other")),
            on="cs_order_number",
        )
        .filter(pl.col("cs_warehouse_sk") != pl.col("other"))
        .select("row_id")
        .unique()
    )
    expected = (
        filtered.with_row_index("row_id")
        .join(exists, on="row_id")
        .join(
            returns.unique(),
            left_on="cs_order_number",
            right_on="cr_order_number",
            how="anti",
        )
        .select(
            pl.col("cs_order_number").n_unique().alias("order count"),
            pl.col("cs_ext_ship_cost").sum().alias("total shipping cost"),
            pl.col("cs_net_profit").sum().alias("total net profit"),
        )
    )
    assert_frame_equal(q.collect(), expected.collect())
    assert_gpu_result_equal(q, engine=engine)
