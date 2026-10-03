from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
)

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.anthropic._types import AttributeValue
from openinference.instrumentation.anthropic._utils import (
    _finish_tracing,
    _get_token_counts,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from httpx2 import Headers

    from anthropic import Stream
    from anthropic.types import RawMessageStreamEvent


class _RawStreamInterceptor(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """
    Wraps the raw HTTP stream inside a MessageStream. Forwards every event
    unchanged so MessageStream can run its own accumulation (accumulate_event),
    and calls _finish_tracing once the stream is exhausted or an error occurs.
    No custom accumulation is needed here because MessageStream.current_message_snapshot
    gives us the complete ParsedMessage at the end.
    """

    __slots__ = ("_self_with_span", "_self_message_stream", "_self_progress")

    def __init__(
        self,
        raw_stream: "Stream[RawMessageStreamEvent]",
        with_span: "_WithSpan",
        message_stream: Any = None,
    ) -> None:
        super().__init__(raw_stream)
        self._self_with_span = with_span
        self._self_message_stream = message_stream
        self._self_progress = _StreamProgress()

    def __iter__(self) -> Iterator["RawMessageStreamEvent"]:
        try:
            for item in self.__wrapped__:
                self._self_progress.process_event(item)
                yield item
        except Exception as exception:
            self._self_with_span.record_exception(exception)
            self._finish_tracing(
                status=trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
            )
            raise
        self._finish_tracing(status=trace_api.Status(status_code=trace_api.StatusCode.OK))

    async def __aiter__(self) -> AsyncIterator["RawMessageStreamEvent"]:
        try:
            async for item in self.__wrapped__:
                self._self_progress.process_event(item)
                yield item
        except Exception as exception:
            self._self_with_span.record_exception(exception)
            self._finish_tracing(
                status=trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
            )
            raise
        self._finish_tracing(status=trace_api.Status(status_code=trace_api.StatusCode.OK))

    def _finish_tracing(self, status: Optional[trace_api.Status] = None) -> None:
        snapshot = None
        if self._self_message_stream is not None:
            try:
                snapshot = self._self_message_stream.current_message_snapshot
            except Exception:
                pass
        _finish_tracing(
            with_span=self._self_with_span,
            has_attributes=_MessageExtractor(snapshot, self._self_progress),
            status=status,
        )


class _MessagesStream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = (
        "_response_accumulator",
        "_with_span",
    )

    def __init__(
        self,
        stream: "Stream[RawMessageStreamEvent]",
        with_span: _WithSpan,
        *,
        is_beta: bool = False,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _MessageResponseAccumulator(
            is_beta=is_beta,
            request_headers=stream.response.request.headers,
        )
        self._with_span = with_span

    # The SDK stream's context manager returns the SDK stream, which would bypass the iteration
    # below, so these return the proxy. Exiting finishes the span if iteration has not, e.g. when
    # the stream is left early, recording the exception that ended the context, e.g. a
    # CancelledError, which iteration does not catch.

    def __enter__(self) -> "_MessagesStream":
        self.__wrapped__.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        except BaseException as exception:
            # e.g. closing the response failed
            self._finish_tracing_on_exit(exception)
            raise
        self._finish_tracing_on_exit(exc_val)

    async def __aenter__(self) -> "_MessagesStream":
        await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        except BaseException as exception:
            # e.g. closing the response failed, or the task was cancelled while it closed
            self._finish_tracing_on_exit(exception)
            raise
        self._finish_tracing_on_exit(exc_val)

    def _finish_tracing_on_exit(self, exception: Optional[BaseException]) -> None:
        # GeneratorExit: a generator holding the context was closed, which leaves the stream
        # early rather than failing the request
        if exception is None or isinstance(exception, GeneratorExit):
            self._finish_tracing()
            return
        self._with_span.record_exception(exception)
        self._finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def __iter__(self) -> Iterator["RawMessageStreamEvent"]:
        try:
            for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    async def __aiter__(self) -> AsyncIterator["RawMessageStreamEvent"]:
        try:
            async for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        _finish_tracing(
            with_span=self._with_span,
            has_attributes=_MessageExtractor(
                self._response_accumulator._result(), self._response_accumulator._progress
            ),
            status=status,
        )


class _MessageResponseAccumulator:
    """Accumulates raw SSE events into a ParsedMessage using the SDK's own accumulate_event."""

    __slots__ = ("_is_beta", "_request_headers", "_snapshot", "_json_bufs", "_progress")

    def __init__(
        self,
        *,
        is_beta: bool,
        request_headers: "Headers",
    ) -> None:
        self._is_beta = is_beta
        self._request_headers = request_headers
        self._snapshot: Any = None
        # Buffers partial tool-use input JSON across events, keyed by content block
        # index.
        self._json_bufs: Dict[int, bytes] = {}
        self._progress = _StreamProgress()

    def process_chunk(self, chunk: "RawMessageStreamEvent") -> None:
        self._progress.process_event(chunk)
        # Beta and stable chunks need their matching accumulate_event; beta's
        # raises on stable chunks and vice versa silently drops updates.
        if self._is_beta:
            from anthropic.lib.streaming._beta_messages import (
                accumulate_event as accumulate_beta_event,
            )

            beta_kwargs: Dict[str, Any] = dict(
                event=chunk,
                current_snapshot=self._snapshot,
                request_headers=self._request_headers,
                json_bufs=self._json_bufs,
            )
            try:
                self._snapshot = accumulate_beta_event(**beta_kwargs)
            except Exception:
                pass
        else:
            from anthropic.lib.streaming._messages import accumulate_event

            try:
                self._snapshot = accumulate_event(
                    event=chunk,
                    current_snapshot=self._snapshot,
                    json_bufs=self._json_bufs,
                )
            except Exception:
                pass

    def _result(self) -> Any:
        return self._snapshot


class _StreamProgress:
    """
    Tracks how far a stream got, which the SDK's snapshot does not tell apart from a finished
    message: a tool call cut off mid-arguments holds only the input that parses so far (e.g. {}
    for '{"city": "Chica'), and usage keeps message_start's output tokens until message_delta
    brings the final count.
    """

    __slots__ = ("unfinished_inputs", "has_final_usage")

    def __init__(self) -> None:
        # Raw input JSON received so far for content blocks not yet stopped, by block index.
        self.unfinished_inputs: Dict[int, str] = {}
        self.has_final_usage = False

    def process_event(self, event: Any) -> None:
        try:
            if event.type == "content_block_start":
                self.unfinished_inputs[event.index] = ""
            elif event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                self.unfinished_inputs[event.index] += event.delta.partial_json
            elif event.type == "content_block_stop":
                self.unfinished_inputs.pop(event.index, None)
            elif event.type == "message_delta":
                self.has_final_usage = True
        except Exception:
            pass


_OUTPUT_TOKEN_COUNTS = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
    SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
)


class _MessageExtractor:
    """
    Extracts span attributes from a ParsedMessage (or Message) snapshot.
    Used by both the messages.stream() path (via current_message_snapshot)
    and the messages.create(stream=True) path (via _MessageResponseAccumulator).
    """

    __slots__ = ("_snapshot", "_progress")

    def __init__(self, snapshot: Any, progress: Optional[_StreamProgress] = None) -> None:
        self._snapshot = snapshot
        self._progress = progress

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        snapshot = self._snapshot
        progress = self._progress
        if snapshot is None:
            return
        yield SpanAttributes.OUTPUT_VALUE, snapshot.model_dump_json()
        yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
        if model_name := getattr(snapshot, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model_name
            yield SpanAttributes.LLM_RESPONSE_MODEL_NAME, model_name
        yield (
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
            snapshot.role,
        )
        if stop_reason := getattr(snapshot, "stop_reason", None):
            yield SpanAttributes.LLM_FINISH_REASON, stop_reason
        tool_idx = 0
        for block_idx, block in enumerate(snapshot.content):
            content_prefix = (
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
                f"{MessageAttributes.MESSAGE_CONTENTS}.{block_idx}"
            )
            if block.type == "text":
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "text",
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    block.text,
                )
            elif block.type == "thinking":
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "reasoning",
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    block.thinking,
                )
                if signature := block.signature:
                    yield (
                        f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}",
                        signature,
                    )
            elif block.type == "redacted_thinking":
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "reasoning",
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_DATA}",
                    block.data,
                )
            elif block.type == "tool_use":
                arguments = safe_json_dumps(block.input)
                if progress and block_idx in progress.unfinished_inputs:
                    # the stream ended inside this tool call, so record the arguments received
                    # rather than the partial parse, which reads as a complete call
                    arguments = progress.unfinished_inputs[block_idx]
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_ID}",
                    block.id,
                )
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    block.name,
                )
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    arguments,
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "tool_use",
                )
                yield (
                    f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_ID}",
                    block.id,
                )
                yield (
                    f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    block.name,
                )
                yield (
                    f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    arguments,
                )
                tool_idx += 1
        for key, value in _get_token_counts(snapshot.usage):
            if progress and not progress.has_final_usage and key in _OUTPUT_TOKEN_COUNTS:
                # the stream ended before message_delta reported the output tokens
                continue
            yield key, value
