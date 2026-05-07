"""OpenTelemetry setup for the multi-agent system.

Provides a single `setup_tracing()` entry point that configures a TracerProvider,
exporter, and the auto-instrumentors needed for Bedrock AgentCore + boto3.

Bedrock AgentCore is BYO-tracer-provider: its `BaggageSpanProcessor` auto-attaches
to whatever provider we set as global on every request, so this module only needs
to install one before the app starts handling traffic.

Exporter selection (driven by env, no code changes between local and AWS):
  - OTEL_TRACES_EXPORTER=otlp     → OTLPSpanExporter (HTTP), reads OTEL_EXPORTER_OTLP_ENDPOINT.
                                     Used in deployed AgentCore Runtime — AWS injects the endpoint.
  - OTEL_TRACES_EXPORTER=console  → ConsoleSpanExporter (default for local dev).
"""

import os

from opentelemetry import trace
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_initialized = False


def setup_tracing(service_name: str = "browser-agent-mas") -> None:
    """Configure global OTel TracerProvider and instrumentors. Idempotent."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter_kind = os.environ.get("OTEL_TRACES_EXPORTER", "console").lower()
    if exporter_kind == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument every boto3 call (Bedrock Converse, Guardrails, etc.).
    BotocoreInstrumentor().instrument()


def instrument_asgi_app(app) -> None:
    """Instrument the AgentCore Starlette app so POST /invocations becomes a root HTTP span."""
    from opentelemetry.instrumentation.starlette import StarletteInstrumentor
    StarletteInstrumentor.instrument_app(app)


def get_tracer(name: str):
    return trace.get_tracer(name)
