import asyncio
import boto3
import yaml
from loguru import logger
from pathlib import Path
from typing import Any, Dict
from opentelemetry import trace

from agent_base import BaseAgent

tracer = trace.get_tracer(__name__)


class WritingAgent(BaseAgent):
    """Agent D — Drafts polished documents from structured analysis using Amazon Bedrock.

    Receives a structured analysis from Agent B and transforms it into a
    well-written document in the requested format and tone.
    """

    SYSTEM_PROMPT = (
        "You are an expert writer and editor. Transform the provided analysis into a polished, "
        "well-structured document. Adapt your tone and output format exactly as instructed. "
        "Produce only the final document — no meta-commentary, no preamble, no explanation of what you are doing."
    )

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.bedrock_client = None

    async def initialize(self) -> None:
        """Load configuration and initialise the Bedrock client."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        region = self.config.get("region", "us-east-1")
        profile = self.config.get("aws_profile", "default")

        try:
            session = boto3.Session(profile_name=profile)
            self.bedrock_client = session.client("bedrock-runtime", region_name=region)
        except Exception as e:
            logger.warning(
                f"WritingAgent: failed to load AWS profile '{profile}', falling back to default credentials: {e}"
            )
            self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

        logger.info("WritingAgent initialised.")

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Draft a polished document from a structured analysis.

        Args:
            request: {
                "analysis": str  — structured analysis to transform (required),
                "format":   str  — output format: "markdown" | "plain" | "html" (optional),
                "tone":     str  — writing tone: "professional" | "casual" | "technical" (optional)
            }
        Returns:
            {"status": "success", "document": str}
            or {"status": "error", "message": str}
        """
        analysis = request.get("analysis", "").strip()
        if not analysis:
            return {
                "status": "error",
                "message": "WritingAgent: received empty analysis — nothing to write.",
            }

        agent_config = self.config.get("writing_agent", {})
        model_id = agent_config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
        bedrock_timeout_s = float(self.config.get("bedrock_timeout_s", 60))
        doc_format = request.get("format", agent_config.get("default_format", "markdown"))
        tone = request.get("tone", agent_config.get("default_tone", "professional"))

        prompt = (
            f"Write a {tone} document in {doc_format} format based on the following analysis:\n\n"
            f"{analysis}"
        )

        with tracer.start_as_current_span("agent.writing.handle") as span:
            span.set_attribute("mas.agent.name", "writing_agent")
            span.set_attribute("gen_ai.system", "aws.bedrock")
            span.set_attribute("gen_ai.request.model", model_id)
            span.set_attribute("mas.format", doc_format)
            span.set_attribute("mas.tone", tone)

            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.bedrock_client.converse,
                        modelId=model_id,
                        system=[{"text": self.SYSTEM_PROMPT}],
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                    ),
                    timeout=bedrock_timeout_s,
                )
                usage = response.get("usage", {}) or {}
                if "inputTokens" in usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage["inputTokens"])
                if "outputTokens" in usage:
                    span.set_attribute("gen_ai.usage.output_tokens", usage["outputTokens"])
                document = response["output"]["message"]["content"][0]["text"]
                logger.info("WritingAgent: document drafted.")
                return {"status": "success", "document": document}
            except asyncio.TimeoutError:
                msg = f"WritingAgent: Bedrock converse timed out after {bedrock_timeout_s:.0f}s"
                logger.error(msg)
                span.set_attribute("mas.status", "timeout")
                return {"status": "error", "message": msg}
            except Exception as e:
                logger.error(f"WritingAgent: Bedrock error: {e}")
                span.record_exception(e)
                return {"status": "error", "message": str(e)}

    async def shutdown(self) -> None:
        """No persistent resources to release (stateless agent)."""
        logger.info("WritingAgent shutdown.")
