import asyncio
import boto3
import yaml
import json
from loguru import logger
from pathlib import Path

from agent_base import BaseAgent
from browser_agent import BrowserAgent

class Orchestrator(BaseAgent):
    """Orchestrator that uses Amazon Bedrock Converse API to manage child agents.

    This local-first prototype uses Bedrock's Converse API to give the model
    access to the BrowserAgent as a tool.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.bedrock_client = None
        self.browser_agent = BrowserAgent(config_path)

    async def initialize(self) -> None:
        """Load configuration and initialise Bedrock client."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                
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

    def _get_tool_config(self):
        """Define the tools available to the model."""
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": "browser_action",
                        "description": "Perform an action in the browser. Supports navigation, clicking, extracting text, and screenshots.",
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
                                        "description": "The CSS selector to interact with (required for 'click' and 'extract')."
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "The file path to save the screenshot (required for 'screenshot')."
                                    }
                                },
                                "required": ["action"]
                            }
                        }
                    }
                }
            ]
        }

    async def handle(self, request: dict) -> dict:
        """Handle a high-level task by conversing with Bedrock and invoking tools."""
        task = request.get("task", "")
        # Use Claude 3 Sonnet by default as it supports tools natively
        model_id = self.config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        logger.info(f"Starting task: {task}")
        
        messages = [{
            "role": "user",
            "content": [{"text": task}]
        }]
        
        tool_config = self._get_tool_config()
        
        # We loop to allow the model to use tools and then answer
        for loop_count in range(5): # Max 5 turns
            logger.info(f"Turn {loop_count + 1} - Calling Bedrock...")
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=messages,
                    toolConfig=tool_config
                )
            except Exception as e:
                logger.error(f"Bedrock API error: {e}")
                return {"status": "error", "message": str(e)}
                
            output_message = response['output']['message']
            messages.append(output_message)
            
            # Check if the model wants to call a tool
            stop_reason = response.get('stopReason')
            if stop_reason == 'tool_use':
                tool_results = []
                for content_block in output_message['content']:
                    if 'toolUse' in content_block:
                        tool_use = content_block['toolUse']
                        tool_name = tool_use['name']
                        tool_input = tool_use['input']
                        tool_use_id = tool_use['toolUseId']
                        
                        logger.info(f"Model requested tool: {tool_name} with input {tool_input}")
                        
                        if tool_name == "browser_action":
                            # Execute the tool locally
                            result = await self.browser_agent.handle(tool_input)
                        else:
                            result = {"error": f"Unknown tool: {tool_name}"}
                            
                        # Format the result for Bedrock
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"json": result}]
                            }
                        })
                        
                # Add tool results to messages and continue the loop
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                # The model finished reasoning
                final_text = next((block['text'] for block in output_message['content'] if 'text' in block), "")
                logger.info("Task completed.")
                return {"status": "success", "response": final_text}
                
        return {"status": "error", "message": "Max turns reached without completing the task"}

    async def shutdown(self) -> None:
        """Shutdown all child agents and clean up resources."""
        await self.browser_agent.shutdown()
        logger.info("Orchestrator shutdown complete")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Orchestrator CLI")
    parser.add_argument("--task", default="Navigate to https://example.com, take a screenshot to example.png, and tell me the page title.", help="The task to perform")
    args = parser.parse_args()
    
    orch = Orchestrator()
    await orch.initialize()
    result = await orch.handle({"task": args.task})
    print(json.dumps(result, indent=2))
    await orch.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
