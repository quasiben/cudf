# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime guard for expressions evaluated on an already complete frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from cudf_polars.dsl.ir import IR

if TYPE_CHECKING:
    from cudf_polars.typing import Schema


class CompleteFrameInput(IR):
    """Permit local frame evaluation only when every rank already owns the input."""

    __slots__ = ()
    _non_child: ClassVar[tuple[str, ...]] = ("schema",)
    _n_non_child_args: ClassVar[int] = 0

    def __init__(self, schema: Schema, child: IR):
        self.schema = schema
        self.children = (child,)
        self._non_child_args = ()
