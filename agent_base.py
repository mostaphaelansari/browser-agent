import abc
from typing import Any, Dict, Literal, TypedDict


class AgentResponse(TypedDict, total=False):
    """Typed response contract for all agents.

    Every ``handle()`` implementation MUST include ``status``.
    Additional keys (e.g. ``analysis``, ``document``, ``text``) are agent-specific.
    """
    status: Literal["success", "error"]
    message: str  # populated on error


class BaseAgent(abc.ABC):
    """Abstract base class for all agents.

    Each agent should implement ``initialize`` to set up any required resources,
    ``handle`` to process an incoming request and return a response, and ``shutdown``
    to clean up.

    Contract: ``handle()`` must always return an ``AgentResponse``-compatible dict
    with at minimum a ``status`` key set to ``"success"`` or ``"error"``.
    """

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Prepare the agent (e.g., load models, start browsers)."""
        pass

    @abc.abstractmethod
    async def handle(self, request: Dict[str, Any]) -> AgentResponse:
        """Process a request and return a response.

        Args:
            request: Arbitrary payload describing the action to perform.
        Returns:
            An AgentResponse dict with at least ``status: "success" | "error"``.
        """
        pass

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Release any held resources (e.g., close browsers)."""
        pass
