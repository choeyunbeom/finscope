"""Critic agent node — citation check + retry decision."""
# TODO Week 2
from .graph import AgentState


async def critic_node(state: AgentState) -> AgentState:
    """Check that >70% of claims have cited source chunks."""
    raise NotImplementedError
