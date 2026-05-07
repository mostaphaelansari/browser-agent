import asyncio
import boto3
import yaml
from loguru import logger
from pathlib import Path
from typing import Any, Dict

from agent_base import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent B — Structures and analyses raw content using Amazon Bedrock.

    Receives raw text from Agent A (Browser) and returns a structured analysis
    with summary, key facts, and notable insights.
    """

    SYSTEM_PROMPT = (
        "You are an expert analyst. When given raw text content, structure it into clear sections: "
        "**Summary**, **Key Facts**, and **Notable Insights**. "
        "Be concise, objective, and factual. Do not add information not present in the source material."
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
                f"AnalysisAgent: failed to load AWS profile '{profile}', falling back to default credentials: {e}"
            )
            self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

        logger.info("AnalysisAgent initialised.")

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse raw text content and return a structured analysis.

        Args:
            request: {
                "content": str  — raw text to analyse (required),
                "focus":   str  — what to focus on, e.g. "pricing", "sentiment" (optional)
            }
        Returns:
            {"status": "success", "analysis": str}
            or {"status": "error", "message": str}
        """
        content = request.get("content", "").strip()
        focus = request.get("focus", "key facts and insights")

        if not content:
            return {
                "status": "error",
                "message": "AnalysisAgent: received empty content — nothing to analyse.",
            }

        agent_config = self.config.get("analysis_agent", {})
        model_id = agent_config.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0")
        bedrock_timeout_s = float(self.config.get("bedrock_timeout_s", 60))

        prompt = f"Focus on: {focus}\n\n---\n\n{content}"

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
            analysis = response["output"]["message"]["content"][0]["text"]
            logger.info("AnalysisAgent: analysis complete.")
            return {"status": "success", "analysis": analysis}
        except asyncio.TimeoutError:
            msg = f"AnalysisAgent: Bedrock converse timed out after {bedrock_timeout_s:.0f}s"
            logger.error(msg)
            return {"status": "error", "message": msg}
        except Exception as e:
            logger.error(f"AnalysisAgent: Bedrock error: {e}")
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> None:
        """No persistent resources to release (stateless agent)."""
        logger.info("AnalysisAgent shutdown.")
