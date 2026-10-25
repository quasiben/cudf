# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for TPC-DS query construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import polars as pl

from cudf_polars.streaming.benchmarks.pdsds_queries import q9, q61
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def run_config():
    return SimpleNamespace(
        dataset_path="unused", suffix="", scale_factor=1000, qualification=False
    )


@pytest.mark.parametrize("empty", [False, True])
def test_q9_count_threshold(engine, monkeypatch, run_config, empty):
    tables = {
        "store_sales": pl.LazyFrame(
            {
                "ss_quantity": [1, 21, 41, 61, 81, None],
                "then": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "else": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
        "reason": pl.LazyFrame({"r_reason_sk": [1]}),
    }
    if empty:
        tables["store_sales"] = tables["store_sales"].head(0)
    monkeypatch.setattr(q9, "get_data", lambda path, name, suffix: tables[name])
    monkeypatch.setattr(
        q9,
        "load_parameters",
        lambda *args, **kwargs: {
            "aggcthen": "then",
            "aggcelse": "else",
            "rc": [0, 1, 0, 1, 0],
        },
    )
    query = q9.polars_impl(run_config).frame
    assert query.collect().row(0) == (
        (None,) * 5 if empty else (10.0, 2.0, 30.0, 4.0, 50.0)
    )
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("empty", [False, True])
def test_q61_row_count(engine, monkeypatch, run_config, empty):
    tables = {
        "store_sales": pl.LazyFrame(
            {
                "ss_store_sk": [1],
                "ss_promo_sk": [1],
                "ss_sold_date_sk": [1],
                "ss_customer_sk": [1],
                "ss_item_sk": [1],
                "ss_ext_sales_price": [10.0],
            }
        ),
        "store": pl.LazyFrame({"s_store_sk": [1], "s_gmt_offset": [0.0]}),
        "promotion": pl.LazyFrame(
            {
                "p_promo_sk": [1],
                "p_channel_dmail": ["Y"],
                "p_channel_email": ["N"],
                "p_channel_tv": ["N"],
            }
        ),
        "date_dim": pl.LazyFrame({"d_date_sk": [1], "d_year": [2000], "d_moy": [1]}),
        "customer": pl.LazyFrame({"c_customer_sk": [1], "c_current_addr_sk": [1]}),
        "customer_address": pl.LazyFrame(
            {"ca_address_sk": [1], "ca_gmt_offset": [0.0]}
        ),
        "item": pl.LazyFrame({"i_item_sk": [1], "i_category": ["Books"]}),
    }
    if empty:
        tables["store_sales"] = tables["store_sales"].head(0)
    monkeypatch.setattr(q61, "get_data", lambda path, name, suffix: tables[name])
    monkeypatch.setattr(
        q61,
        "load_parameters",
        lambda *args, **kwargs: {
            "year": 2000,
            "month": 1,
            "gmt_offset": 0.0,
            "category": "Books",
        },
    )
    query = q61.polars_impl(run_config).frame
    assert query.collect().row(0) == ((None,) * 3 if empty else (10.0, 10.0, 100.0))
    assert_gpu_result_equal(query, engine=engine)
