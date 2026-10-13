# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check complete-frame ownership before native non-pointwise selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.complete_frame import CompleteFrameInput
from cudf_polars.streaming.utils import _fallback_inform

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@define_actor()
async def complete_frame_input_node(
    context: Context,
    comm: Communicator,
    ir: CompleteFrameInput,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    config_options: ConfigOptions[StreamingExecutor],
) -> None:
    """Forward complete input; enforce fallback policy before any remote gather."""
    async with shutdown_on_error(
        context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir, ir_context=ir_context
    ) as scope:
        metadata = await recv_metadata(ch_in, context)
        complete = metadata.duplicated or comm.nranks == 1
        if not complete:
            _fallback_inform(
                "This selection is not supported for multiple partitions: "
                "its input is not already complete on every rank.",
                config_options,
            )
        if scope.tracer is not None:
            scope.tracer.decision = "complete_input" if complete else "fallback_gather"
            if metadata.duplicated:
                scope.tracer.set_duplicated()
        await send_metadata(ch_out, context, metadata)
        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(msg, br=context.br())
            await send_chunk(
                context, ch_out, chunk, msg.sequence_number, tracer=scope.tracer
            )
        await ch_out.drain(context)


@generate_ir_sub_network.register(CompleteFrameInput)
def _(
    ir: CompleteFrameInput, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        complete_frame_input_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            rec.state["config_options"],
        )
    ]
    return nodes, channels
