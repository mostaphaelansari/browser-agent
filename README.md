# Bedrock Browser Multi-Agent System

A local-first prototype demonstrating a multi-agent orchestration pattern using the Amazon Bedrock Converse API. This project implements an **Orchestrator Agent** that communicates with a foundation model (e.g., Amazon Nova Lite) and routes requested tool actions to a **Browser Agent** powered by Playwright.

## Features

- **Bedrock Converse API Integration**: Leverages native tool-calling features of Bedrock models to perform reasoning and actions.
- **Async Browser Automation**: Uses Playwright to asynchronously launch a headless browser, navigate, extract text, and take screenshots.
- **Extensible Architecture**: The `BaseAgent` abstraction allows you to easily add new local agents and tools.

## Prerequisites

- Python 3.10+
- AWS CLI configured with active credentials (`~/.aws/credentials`)
- Access to Amazon Bedrock foundation models (e.g. Amazon Nova Lite) in your chosen region.

## Setup

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright Browsers**:
   ```bash
   playwright install chromium
   ```

4. **AWS Credentials**:
   Ensure you have configured your AWS credentials either via `aws configure` (recommended) or an `.env` file. The AWS User/Role must have permission to call `bedrock:InvokeModel` and `bedrock:Converse`.

## Configuration

Edit `config.yaml` to change settings such as the AWS region, model ID, or Playwright headless mode:

```yaml
region: eu-central-1
model_id: eu.amazon.nova-lite-v1:0
headless: true
browser_timeout: 30
```

## Usage

Run the orchestrator with a task description. The orchestrator will converse with Bedrock, invoke the browser tool locally when requested by the model, and return the final answer.

```bash
python orchestrator.py --task "Navigate to https://example.com, take a screenshot named example.png, and tell me the main heading text."
```

## Project Structure

- `orchestrator.py`: The main entry point. Sets up the Bedrock runtime and coordinates tool calls.
- `browser_agent.py`: A Playwright-powered agent that exposes browser actions as tools to the orchestrator.
- `agent_base.py`: The abstract base class defining the agent lifecycle (`initialize`, `handle`, `shutdown`).
- `config.yaml`: Core configurations for region, models, and tools.
