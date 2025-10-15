"""
Planner Adapter - Main Agent for Ideas and Hypotheses

This adapter wraps IdeaGenerationService for main agent orchestration.
It only generates ideas and hypotheses using LLM - NO code execution.
"""

from .adapter import PlannerAdapter

__all__ = ['PlannerAdapter']
