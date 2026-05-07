import abc
from typing import Any, Dict

class BaseAgent(abc.ABC):
    """Abstract base class for all agents.

    Each agent should implement ``initialize`` to set up any required resources,
    ``handle`` to process an incoming request and return a response, and ``shutdown``
    to clean up.
    """

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Prepare the agent (e.g., load models, start browsers)."""
        pass

    @abc.abstractmethod
    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return a response.

        Args:
            request: Arbitrary payload describing the action to perform.
        Returns:
            A dictionary containing the response data.
        """
        pass

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Release any held resources (e.g., close browsers)."""
        pass
