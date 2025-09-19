"""
README.md Generator for uagent Research Tasks
Automatically generates comprehensive documentation after successful experiments
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .memory_manager import memory_manager
from .llm_client import llm_client

logger = logging.getLogger(__name__)

class ReadmeGenerator:
    """Generates comprehensive README.md files based on experiment results and memory"""

    def __init__(self):
        self.memory = memory_manager

    async def generate_experiment_readme(
        self,
        goal_id: str,
        workspace_path: str,
        goal_title: str,
        goal_description: str,
        experiment_results: Dict[str, Any],
        success_criteria: List[str] = None
    ) -> str:
        """Generate a detailed README using LLM based on actual experiment results and workspace content"""

        # Get workspace file contents and structure
        workspace_analysis = self._analyze_workspace_content(workspace_path)

        # Get experiment memory context
        memories = await self.memory.retrieve_memories(goal_id=goal_id, limit=20)

        # Create prompt for LLM to generate comprehensive README
        readme_prompt = self._create_readme_prompt(
            goal_title=goal_title,
            goal_description=goal_description,
            workspace_analysis=workspace_analysis,
            experiment_results=experiment_results,
            memories=memories,
            success_criteria=success_criteria
        )

        try:
            # Call LLM to generate detailed README
            response = await llm_client.chat_completion([
                {"role": "system", "content": "You are an expert technical writer and researcher. Generate a comprehensive, professional README.md that documents what was actually accomplished in this research experiment."},
                {"role": "user", "content": readme_prompt}
            ])

            readme_content = response.choices[0].message.content

            # Write README.md to workspace
            readme_path = Path(workspace_path) / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            logger.info(f"Generated experiment README.md at {readme_path}")

            # Store README generation in memory
            await self.memory.store_memory(
                goal_id=goal_id,
                node_id="readme_generator",
                entry_type="success",
                title="Experiment README.md Generated",
                content=f"LLM-generated comprehensive README.md documenting experiment results and reproduction steps",
                context={
                    "readme_path": str(readme_path),
                    "word_count": len(readme_content.split()),
                    "generation_method": "llm_generated"
                },
                workspace_path=workspace_path
            )

            return readme_content

        except Exception as e:
            logger.error(f"Failed to generate experiment README.md: {e}")
            # Fallback to memory-based generation
            return await self.generate_readme(goal_id, workspace_path, goal_title, goal_description, success_criteria)

    async def generate_readme(
        self,
        goal_id: str,
        workspace_path: str,
        goal_title: str,
        goal_description: str,
        success_criteria: List[str] = None
    ) -> str:
        """Generate a comprehensive README.md for a research goal"""

        # Get comprehensive memory summary
        summary = await self.memory.get_goal_summary(goal_id)

        # Get specific memories for different sections
        successes = await self.memory.retrieve_memories(
            goal_id=goal_id,
            entry_type="success",
            limit=10
        )

        experiments = await self.memory.retrieve_memories(
            goal_id=goal_id,
            entry_type="experiment",
            limit=15
        )

        findings = await self.memory.retrieve_memories(
            goal_id=goal_id,
            entry_type="finding",
            limit=10
        )

        errors = await self.memory.retrieve_memories(
            goal_id=goal_id,
            entry_type="error",
            limit=5
        )

        # Generate README content
        readme_content = self._build_readme_content(
            goal_title=goal_title,
            goal_description=goal_description,
            goal_id=goal_id,
            workspace_path=workspace_path,
            success_criteria=success_criteria or [],
            summary=summary,
            successes=successes,
            experiments=experiments,
            findings=findings,
            errors=errors
        )

        # Write README.md to workspace
        readme_path = Path(workspace_path) / "README.md"
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info(f"Generated README.md at {readme_path}")

            # Store README generation in memory
            await self.memory.store_memory(
                goal_id=goal_id,
                node_id="readme_generator",
                entry_type="success",
                title="README.md Generated",
                content=f"Comprehensive README.md generated with {len(successes)} successes, {len(experiments)} experiments, and {len(findings)} findings",
                context={
                    "readme_path": str(readme_path),
                    "total_sections": 8,
                    "word_count": len(readme_content.split())
                },
                workspace_path=workspace_path
            )

        except Exception as e:
            logger.error(f"Failed to write README.md: {e}")
            await self.memory.store_memory(
                goal_id=goal_id,
                node_id="readme_generator",
                entry_type="error",
                title="README.md Generation Failed",
                content=f"Failed to generate README.md: {str(e)}",
                workspace_path=workspace_path
            )

        return readme_content

    def _build_readme_content(
        self,
        goal_title: str,
        goal_description: str,
        goal_id: str,
        workspace_path: str,
        success_criteria: List[str],
        summary: Dict[str, Any],
        successes: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        errors: List[Dict[str, Any]]
    ) -> str:
        """Build the complete README.md content"""

        content = []

        # Header
        content.append(f"# {goal_title}")
        content.append("")
        content.append(f"> **Research Goal ID:** `{goal_id}`")
        content.append(f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"> **Workspace:** `{workspace_path}`")
        content.append("")

        # Overview
        content.append("## 📋 Overview")
        content.append("")
        content.append(goal_description)
        content.append("")

        # Success Criteria
        if success_criteria:
            content.append("## 🎯 Success Criteria")
            content.append("")
            for i, criteria in enumerate(success_criteria, 1):
                content.append(f"{i}. {criteria}")
            content.append("")

        # Executive Summary
        content.append("## 📊 Executive Summary")
        content.append("")
        content.append(f"- **Total Experiments:** {summary['total_entries']}")
        content.append(f"- **Successful Outcomes:** {len(summary.get('successes', []))}")
        content.append(f"- **Key Findings:** {len(summary.get('findings', []))}")
        content.append(f"- **Errors Encountered:** {len(summary.get('errors', []))}")
        content.append(f"- **Workspace Directories:** {len(summary.get('workspace_paths', []))}")
        content.append("")

        # Key Achievements
        if successes:
            content.append("## ✅ Key Achievements")
            content.append("")
            for success in successes[:5]:  # Top 5 successes
                content.append(f"### {success['title']}")
                content.append("")
                content.append(success['content'])
                content.append("")
                if success.get('context'):
                    try:
                        context = success['context'] if isinstance(success['context'], dict) else {}
                        if context:
                            content.append("**Technical Details:**")
                            for key, value in context.items():
                                if key not in ['timestamp', 'internal_id']:
                                    content.append(f"- **{key.replace('_', ' ').title()}:** {value}")
                            content.append("")
                    except:
                        pass

        # Experimental Results
        if experiments:
            content.append("## 🧪 Experimental Results")
            content.append("")
            content.append("| Experiment | Type | Outcome | Key Learning |")
            content.append("|------------|------|---------|--------------|")

            for exp in experiments[:10]:  # Top 10 experiments
                title = exp['title'][:30] + "..." if len(exp['title']) > 30 else exp['title']
                exp_type = exp.get('entry_type', 'experiment')
                content_preview = exp['content'][:50] + "..." if len(exp['content']) > 50 else exp['content']
                content.append(f"| {title} | {exp_type} | ✅ | {content_preview} |")

            content.append("")

        # Technical Findings
        if findings:
            content.append("## 🔍 Technical Findings")
            content.append("")
            for finding in findings[:8]:  # Top 8 findings
                content.append(f"### {finding['title']}")
                content.append("")
                content.append(finding['content'])
                content.append("")

        # Implementation Details
        content.append("## 🛠 Implementation Details")
        content.append("")

        # Check for common file types and provide installation/usage instructions
        workspace_files = self._get_workspace_files(workspace_path)

        if any(f.endswith('.py') for f in workspace_files):
            content.append("### Python Implementation")
            content.append("")
            content.append("```bash")
            content.append("# Install dependencies")
            content.append("pip install -r requirements.txt")
            content.append("")
            content.append("# Run the implementation")
            content.append("python main.py")
            content.append("```")
            content.append("")

        if any(f.endswith('Dockerfile') for f in workspace_files):
            content.append("### Docker Implementation")
            content.append("")
            content.append("```bash")
            content.append("# Build the Docker image")
            content.append("docker build -t research-project .")
            content.append("")
            content.append("# Run the container")
            content.append("docker run -p 8080:8080 research-project")
            content.append("```")
            content.append("")

        if any(f.endswith('.yml') or f.endswith('.yaml') for f in workspace_files):
            content.append("### Kubernetes/Docker Compose")
            content.append("")
            content.append("```bash")
            content.append("# Deploy using Docker Compose")
            content.append("docker-compose up -d")
            content.append("")
            content.append("# Or deploy to Kubernetes")
            content.append("kubectl apply -f deployment.yaml")
            content.append("```")
            content.append("")

        # File Structure
        if workspace_files:
            content.append("### File Structure")
            content.append("")
            content.append("```")
            for file in sorted(workspace_files)[:20]:  # Show first 20 files
                content.append(file)
            if len(workspace_files) > 20:
                content.append(f"... and {len(workspace_files) - 20} more files")
            content.append("```")
            content.append("")

        # Issues and Solutions
        if errors:
            content.append("## ⚠️ Issues and Solutions")
            content.append("")
            for error in errors[:5]:  # Top 5 errors
                content.append(f"### Issue: {error['title']}")
                content.append("")
                content.append(f"**Problem:** {error['content']}")
                content.append("")
                # Try to find corresponding solutions
                solution_memories = [
                    s for s in successes
                    if any(keyword in s['content'].lower()
                          for keyword in error['title'].lower().split()[:3])
                ]
                if solution_memories:
                    content.append(f"**Solution:** {solution_memories[0]['content']}")
                    content.append("")

        # Next Steps
        content.append("## 🚀 Next Steps")
        content.append("")
        content.append("Based on the experimental results, consider these next steps:")
        content.append("")

        # Generate next steps based on findings
        if summary.get('key_learnings'):
            for learning in summary['key_learnings'][:3]:
                content.append(f"1. **{learning['title']}** - {learning['content'][:100]}...")
        else:
            content.append("1. **Performance Optimization** - Review and optimize the implementation for production use")
            content.append("2. **Testing and Validation** - Implement comprehensive testing suite")
            content.append("3. **Documentation** - Expand technical documentation and user guides")
            content.append("4. **Monitoring** - Add logging and monitoring capabilities")

        content.append("")

        # Footer
        content.append("## 📝 Research Metadata")
        content.append("")
        content.append("| Metric | Value |")
        content.append("|--------|-------|")
        content.append(f"| Research Goal ID | `{goal_id}` |")
        content.append(f"| Total Memory Entries | {summary['total_entries']} |")
        content.append(f"| Workspace Path | `{workspace_path}` |")
        content.append(f"| Generation Time | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
        content.append(f"| uagent Version | 1.0.0 |")
        content.append("")
        content.append("---")
        content.append("")
        content.append("*This README.md was automatically generated by uagent research tree system based on experimental results and accumulated knowledge.*")

        return "\n".join(content)

    def _analyze_workspace_content(self, workspace_path: str) -> Dict[str, Any]:
        """Analyze workspace content including files, code, and structure"""
        workspace = Path(workspace_path)
        analysis = {
            "file_structure": [],
            "code_files": {},
            "data_files": [],
            "output_files": [],
            "total_files": 0
        }

        if not workspace.exists():
            return analysis

        try:
            # Get file structure
            for file_path in workspace.rglob('*'):
                if file_path.is_file():
                    rel_path = file_path.relative_to(workspace)
                    analysis["file_structure"].append(str(rel_path))
                    analysis["total_files"] += 1

                    # Categorize files
                    if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if len(content) < 10000:  # Only include smaller files
                                    analysis["code_files"][str(rel_path)] = content
                        except:
                            pass
                    elif file_path.suffix in ['.json', '.csv', '.txt', '.log']:
                        analysis["data_files"].append(str(rel_path))
                    elif 'output' in str(file_path).lower():
                        analysis["output_files"].append(str(rel_path))

        except Exception as e:
            logger.error(f"Failed to analyze workspace content: {e}")

        return analysis

    def _create_readme_prompt(
        self,
        goal_title: str,
        goal_description: str,
        workspace_analysis: Dict[str, Any],
        experiment_results: Dict[str, Any],
        memories: List[Dict[str, Any]],
        success_criteria: List[str] = None
    ) -> str:
        """Create a comprehensive prompt for LLM to generate README"""

        prompt = f"""Generate a comprehensive, professional README.md for this research experiment.

# Research Goal
**Title:** {goal_title}
**Description:** {goal_description}

# Success Criteria
{chr(10).join(f"- {criteria}" for criteria in (success_criteria or []))}

# Experiment Results
{json.dumps(experiment_results, indent=2)}

# Workspace Analysis
**Total Files:** {workspace_analysis.get('total_files', 0)}
**File Structure:**
{chr(10).join(f"- {f}" for f in workspace_analysis.get('file_structure', [])[:20])}

# Code Files Generated:
"""

        # Include code snippets
        for file_path, content in workspace_analysis.get('code_files', {}).items():
            prompt += f"\n## {file_path}\n```\n{content[:1000]}...\n```\n"

        prompt += f"""

# Experiment Memory Context
Recent experiment activities:
"""

        # Include relevant memories
        for memory in memories[:10]:
            prompt += f"\n- **{memory.get('title', 'Unknown')}**: {memory.get('content', '')[:200]}..."

        prompt += f"""

# README Requirements

Generate a comprehensive README.md that includes:

1. **Project Overview** - Clear description of what was accomplished
2. **What Was Done** - Detailed explanation of the experimental process and results
3. **How to Reproduce** - Step-by-step instructions to reproduce the experiment
4. **File Structure** - Explanation of generated files and their purposes
5. **Key Findings** - Important discoveries and outcomes
6. **Installation & Usage** - How to install dependencies and run the code
7. **Results Analysis** - Analysis of what worked and what didn't
8. **Next Steps** - Recommendations for future work

Focus on documenting what was ACTUALLY done in this experiment, not generic templates. Include specific code examples, command-line instructions, and detailed reproduction steps.

Use proper markdown formatting with code blocks, tables, and clear section headers.
"""

        return prompt

    def _get_workspace_files(self, workspace_path: str) -> List[str]:
        """Get list of files in workspace directory"""
        try:
            workspace = Path(workspace_path)
            if workspace.exists():
                files = []
                for file_path in workspace.rglob('*'):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(workspace)
                        files.append(str(rel_path))
                return files
        except Exception as e:
            logger.error(f"Failed to get workspace files: {e}")
        return []

# Global instance
readme_generator = ReadmeGenerator()