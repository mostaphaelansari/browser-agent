import asyncio
import os
import boto3
import yaml
import json
from loguru import logger
from pathlib import Path
from contextlib import asynccontextmanager, AsyncExitStack
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OTel must be configured before BedrockAgentCoreApp is constructed so its
# BaggageSpanProcessor attaches to our provider (and before any boto3 client
# is created so BotocoreInstrumentor can patch it).
from tracing import setup_tracing, instrument_asgi_app, get_tracer
setup_tracing(service_name="browser-agent-mas")

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from agent_base import BaseAgent
from browser_agent import BrowserAgent
from analysis_agent import AnalysisAgent
from writing_agent import WritingAgent

tracer = get_tracer(__name__)

# System prompt that enforces the Browser → Analyse → Rédige pipeline order.
ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are an orchestrator managing three specialized agents for research and writing tasks. "
    "Always follow this strict pipeline order:\n"
    "1. Use 'browser_action' (navigate, then extract) to collect raw content from the web.\n"
    "2. Use 'analysis_action' to structure and analyse the collected content. Never skip this step.\n"
    "3. Use 'writing_action' to produce the final polished document.\n"
    "If any step returns an error status, report it clearly and stop. "
    "Do not proceed to the next agent with empty or failed input. "
    "Return plain text only. Do not emit <thinking> or <response> tags."
)

# Global orchestrator instance managed by the AgentCore lifespan
orch: "Orchestrator" = None  # type: ignore


class Orchestrator(BaseAgent):
    """Orchestrator that uses Amazon Bedrock Converse API to manage child agents.

    This local-first prototype uses Bedrock's Converse API to give the model
    access to the BrowserAgent as a tool, as well as dynamic MCP tools.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.bedrock_client = None
        # Child agents
        self.browser_agent = BrowserAgent(config_path)
        self.analysis_agent = AnalysisAgent(config_path)
        self.writing_agent = WritingAgent(config_path)
        # MCP
        self.exit_stack = AsyncExitStack()
        self.mcp_sessions: dict = {}  # name -> session
        self.mcp_tools: list = []     # translated tool specs
        self.mcp_tool_map: dict = {}  # tool_name -> server_name

    def _is_stateless(self) -> bool:
        """Stateless = no persistent memory / external state tools (e.g. sqlite MCP)."""
        memory = self.config.get("memory", {}) if isinstance(self.config, dict) else {}
        if isinstance(memory, dict):
            mode = memory.get("mode")
        else:
            mode = None
        if mode is None:
            mode = self.config.get("memory_mode")
        return str(mode).lower() in {"stateless", "none", "off", "disabled", "false", "0"}

    async def initialize(self) -> None:
        """Load configuration, initialise Bedrock client and MCP servers."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        if not isinstance(self.config, dict):
            self.config = {}

        stateless = self._is_stateless()
                
        region = self.config.get("region", "us-east-1")
        profile = self.config.get("aws_profile", "default")
        
        try:
            session = boto3.Session(profile_name=profile)
            self.bedrock_client = session.client("bedrock-runtime", region_name=region)
        except Exception as e:
            logger.warning(f"Failed to load AWS profile '{profile}', falling back to default credentials: {e}")
            self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            
        logger.info(f"Orchestrator initialised with Bedrock Runtime in {region}")
        await self.browser_agent.initialize()
        await self.analysis_agent.initialize()
        await self.writing_agent.initialize()
        
        # Initialize MCP Servers
        mcp_servers = {} if stateless else (self.config.get("mcp_servers", {}) or {})
        for name, srv_config in mcp_servers.items():
            command = srv_config.get("command")
            args = srv_config.get("args", [])
            env = srv_config.get("env", None)
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            try:
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.mcp_sessions[name] = session
                
                # Fetch tools
                response = await session.list_tools()
                for tool in response.tools:
                    # Translate MCP tool to Bedrock toolConfig
                    bedrock_tool = {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": {
                                "json": tool.inputSchema
                            }
                        }
                    }
                    self.mcp_tools.append(bedrock_tool)
                    self.mcp_tool_map[tool.name] = name
                logger.info(f"Connected to MCP server '{name}' and loaded tools")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{name}': {e}")

        if stateless:
            logger.info("Stateless memory mode: MCP servers disabled.")

    def _get_tool_config(self) -> dict:
        """Define the tools available to the model (Browser + Analysis + Writing + MCP)."""
        browser_tool = {
            "toolSpec": {
                "name": "browser_action",
                "description": "Perform an action in the browser. Use this first to collect raw web content.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["navigate", "click", "extract", "screenshot"],
                                "description": "The action to perform in the browser."
                            },
                            "url": {
                                "type": "string",
                                "description": "The URL to navigate to (required for 'navigate')."
                            },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector to interact with (required for 'click' and 'extract')."
                            },
                            "path": {
                                "type": "string",
                                "description": "File path to save the screenshot (required for 'screenshot')."
                            }
                        },
                        "required": ["action"]
                    }
                }
            }
        }

        analysis_tool = {
            "toolSpec": {
                "name": "analysis_action",
                "description": (
                    "Analyse and structure raw text content into summary, key facts, and insights. "
                    "Use this after browser_action to process collected content."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Raw text content to analyse (required)."
                            },
                            "focus": {
                                "type": "string",
                                "description": "What to focus on, e.g. 'pricing', 'key facts', 'sentiment' (optional)."
                            }
                        },
                        "required": ["content"]
                    }
                }
            }
        }

        writing_tool = {
            "toolSpec": {
                "name": "writing_action",
                "description": (
                    "Draft a polished document from a structured analysis. "
                    "Use this as the final step to produce the output."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "Structured analysis to transform into a document (required)."
                            },
                            "format": {
                                "type": "string",
                                "enum": ["markdown", "plain", "html"],
                                "description": "Output format (default: markdown)."
                            },
                            "tone": {
                                "type": "string",
                                "enum": ["professional", "casual", "technical"],
                                "description": "Writing tone (default: professional)."
                            }
                        },
                        "required": ["analysis"]
                    }
                }
            }
        }

        tools = [browser_tool, analysis_tool, writing_tool] + self.mcp_tools
        return {"tools": tools}

    def _scrub_content(self, content: str) -> str:
        """Apply Bedrock Guardrails to scraped content if a guardrail_id is configured.

        Protects against indirect prompt injection from malicious web pages.
        If no guardrail is configured, the content is returned as-is.
        """
        guardrail_id = self.config.get("guardrail_id", "").strip()
        if not guardrail_id:
            return content

        guardrail_version = self.config.get("guardrail_version", "DRAFT")
        try:
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion=guardrail_version,
                source="INPUT",
                content=[{"text": {"text": content}}],
            )
            if response.get("action") == "GUARDRAIL_INTERVENED":
                logger.warning("Guardrail intervened on scraped content — replacing with safe placeholder.")
                return "[Content blocked by Bedrock Guardrail]"
            outputs = response.get("output", [])
            if outputs:
                return outputs[0].get("text", {}).get("text", content)
        except Exception as e:
            logger.error(f"Guardrail check failed (content passed through unmodified): {e}")
        return content

    async def handle(self, request: dict) -> dict:
        """Handle a high-level task by conversing with Bedrock and invoking the agent pipeline.

        Pipeline enforced by system prompt: browser_action → analysis_action → writing_action.
        """
        task = request.get("task", "")
        model_id = self.config.get("model_id", "eu.amazon.nova-lite-v1:0")
        max_turns = int(self.config.get("max_turns", 15))
        bedrock_timeout_s = float(self.config.get("bedrock_timeout_s", 60))

        logger.info(f"Starting task: {task}")

        messages = [{"role": "user", "content": [{"text": task}]}]
        tool_config = self._get_tool_config()

        for loop_count in range(max_turns):
            with tracer.start_as_current_span("mas.react.turn") as turn_span:
                turn_span.set_attribute("mas.turn", loop_count + 1)
                turn_span.set_attribute("mas.turn.max", max_turns)
                logger.info(f"Turn {loop_count + 1}/{max_turns} — Calling Bedrock Orchestrator...")
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.bedrock_client.converse,
                            modelId=model_id,
                            system=[{"text": ORCHESTRATOR_SYSTEM_PROMPT}],
                            messages=messages,
                            toolConfig=tool_config,
                        ),
                        timeout=bedrock_timeout_s,
                    )
                except asyncio.TimeoutError:
                    msg = f"Bedrock converse timed out after {bedrock_timeout_s:.0f}s"
                    logger.error(msg)
                    turn_span.set_attribute("mas.status", "timeout")
                    return {"status": "error", "message": msg}
                except Exception as e:
                    logger.error(f"Bedrock API error: {e}")
                    turn_span.set_attribute("mas.status", "error")
                    turn_span.record_exception(e)
                    return {"status": "error", "message": str(e)}

                # GenAI semantic conventions: emit token usage for cost / dashboarding.
                usage = response.get("usage", {}) or {}
                if "inputTokens" in usage:
                    turn_span.set_attribute("gen_ai.usage.input_tokens", usage["inputTokens"])
                if "outputTokens" in usage:
                    turn_span.set_attribute("gen_ai.usage.output_tokens", usage["outputTokens"])
                turn_span.set_attribute("gen_ai.system", "aws.bedrock")
                turn_span.set_attribute("gen_ai.request.model", model_id)

                output_message = response["output"]["message"]
                messages.append(output_message)

                stop_reason = response.get("stopReason")
                turn_span.set_attribute("mas.stop_reason", stop_reason or "unknown")

                if stop_reason == "tool_use":
                    tool_results = []
                    for content_block in output_message["content"]:
                        if "toolUse" not in content_block:
                            continue

                        tool_use = content_block["toolUse"]
                        tool_name = tool_use["name"]
                        tool_input = tool_use["input"]
                        tool_use_id = tool_use["toolUseId"]

                        logger.info(f"Tool call: {tool_name} | input: {json.dumps(tool_input)[:200]}")

                        with tracer.start_as_current_span("mas.tool_dispatch") as dispatch_span:
                            dispatch_span.set_attribute("mas.tool.name", tool_name)
                            dispatch_span.set_attribute("mas.tool.use_id", tool_use_id)
                            sub_action = tool_input.get("action")
                            if sub_action:
                                dispatch_span.set_attribute("mas.tool.action", sub_action)

                            # ── Agent A: Browser ─────────────────────────────────────
                            if tool_name == "browser_action":
                                result = await self.browser_agent.handle(tool_input)
                                # Scrub any text fields to prevent prompt injection
                                if result.get("status") == "success":
                                    for field in ("text", "title"):
                                        if field in result:
                                            result[field] = self._scrub_content(str(result[field]))

                            # ── Agent B: Analysis ────────────────────────────────────
                            elif tool_name == "analysis_action":
                                if not tool_input.get("content", "").strip():
                                    result = {"status": "error", "message": "analysis_action: 'content' is empty — did browser_action succeed?"}
                                else:
                                    result = await self.analysis_agent.handle(tool_input)

                            # ── Agent D: Writing ─────────────────────────────────────
                            elif tool_name == "writing_action":
                                if not tool_input.get("analysis", "").strip():
                                    result = {"status": "error", "message": "writing_action: 'analysis' is empty — did analysis_action succeed?"}
                                else:
                                    result = await self.writing_agent.handle(tool_input)

                            # ── MCP Tools ────────────────────────────────────────────
                            elif tool_name in self.mcp_tool_map:
                                server_name = self.mcp_tool_map[tool_name]
                                session = self.mcp_sessions[server_name]
                                try:
                                    mcp_result = await session.call_tool(tool_name, tool_input)
                                    result = {"status": "success", "content": [c.model_dump() for c in mcp_result.content]}
                                    if mcp_result.isError:
                                        result["status"] = "error"
                                except Exception as e:
                                    result = {"status": "error", "message": str(e)}

                            else:
                                result = {"status": "error", "message": f"Unknown tool: {tool_name}"}

                            dispatch_span.set_attribute("mas.tool.status", result.get("status", "unknown"))

                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"json": result}],
                            }
                        })

                    messages.append({"role": "user", "content": tool_results})

                else:
                    # Model finished reasoning — extract final text
                    final_text = next(
                        (block["text"] for block in output_message["content"] if "text" in block), ""
                    )
                    # Some models may wrap output in <thinking>/<response>. Strip to user-facing response.
                    if "<response>" in final_text and "</response>" in final_text:
                        final_text = final_text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
                    logger.info("Task completed successfully.")
                    return {"status": "success", "response": final_text}

        return {"status": "error", "message": f"Max turns ({max_turns}) reached without completing the task."}

    async def shutdown(self) -> None:
        """Shutdown all child agents and clean up resources."""
        await self.exit_stack.aclose()
        await self.browser_agent.shutdown()
        await self.analysis_agent.shutdown()
        await self.writing_agent.shutdown()
        logger.info("Orchestrator shutdown complete.")

@asynccontextmanager
async def lifespan(app):
    """AgentCore lifespan: initialise all resources on startup, clean up on shutdown."""
    global orch
    orch = Orchestrator()
    await orch.initialize()
    logger.info("Orchestrator ready — AgentCore HTTP server starting.")
    yield
    await orch.shutdown()
    logger.info("Orchestrator shutdown complete.")


app = BedrockAgentCoreApp(lifespan=lifespan)
instrument_asgi_app(app)


@app.entrypoint
async def invoke(payload: dict) -> dict:
    """Main entrypoint called by AgentCore for every incoming request."""
    task = payload.get("prompt") or payload.get("task", "")
    with tracer.start_as_current_span("mas.invocation") as span:
        span.set_attribute("mas.task.length", len(task))
        try:
            result = await orch.handle({"task": task})
            span.set_attribute("mas.status", result.get("status", "unknown"))
            response_text = result.get("response", "")
            if response_text:
                span.set_attribute("mas.response.length", len(response_text))
            return result
        finally:
            # AgentCore creates a new event loop per request; Playwright objects
            # bound to a previous request's loop hang silently. Tear down the
            # browser here so the next request starts fresh on its own loop.
            await orch.browser_agent.shutdown()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port_raw = os.environ.get("PORT", "8080")
    try:
        port = int(port_raw)
    except ValueError:
        port = 8080

    # BedrockAgentCoreApp.run() may or may not accept host/port depending on version.
    try:
        app.run(host=host, port=port)
    except TypeError:
        logger.warning("AgentCore runtime does not support host/port args; falling back to default app.run().")
        app.run()
