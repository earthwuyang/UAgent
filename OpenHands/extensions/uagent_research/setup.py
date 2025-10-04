"""Setup configuration for UAgent Research Extension"""

from setuptools import setup, find_packages

setup(
    name="uagent-research-extension",
    version="0.1.0",
    description="Advanced research capabilities for OpenHands",
    author="UAgent Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # OpenHands is not needed as a dependency when running within OpenHands
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.29.0",  # For PostgreSQL async support
        "aiosqlite>=0.19.0",  # For SQLite async support
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",
        ]
    },
    entry_points={
        "openhands.extensions": [
            "uagent_research = uagent_research:UAgentResearchExtension",
        ],
        "openhands.agents": [
            "scientific_research = uagent_research.agents:ScientificResearchAgent",
            "code_research = uagent_research.agents:CodeResearchAgent",
        ]
    },
)
