# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.dsl.ir import DataFrameScan
from cudf_polars.dsl.translate import Translator


@pytest.mark.parametrize("version", [(15, 2), (16, 0), (16, 1), (17, 0)])
def test_ir_version(version):
    visitor = pl.LazyFrame({"a": [1]})._ldf.visit()

    class VersionedVisitor:
        def version(self):
            return version

        def __getattr__(self, name):
            return getattr(visitor, name)

    translator = Translator(VersionedVisitor(), pl.GPUEngine(raise_on_fail=True))
    if version <= (16, 0):
        assert isinstance(translator.translate_ir(), DataFrameScan)
        assert not translator.errors
    else:
        with pytest.raises(NotImplementedError, match="No support for polars IR"):
            translator.translate_ir()
