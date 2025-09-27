"""Smart Router API endpoints"""

import asyncio
import logging
from dataclasses import asdict
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..core.app_state import get_app_state
from ..core.smart_router import ClassificationRequest, ClassificationResult
from ..core.websocket_manager import progress_tracker
from ..core.streaming_llm_client import StreamingLLMClient
from ..core.openhands.streaming_integration import enable_openhands_streaming


logger = logging.getLogger(__name__)
router = APIRouter()


class RouterRequest(BaseModel):
    """Smart router request model"""
    user_request: str = Field(..., description="User request to classify and route")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    context: Optional[Dict[str, Any]] = Field({}, description="Additional context")


class RouterResponse(BaseModel):
    """Smart router response model"""
    primary_engine: str = Field(..., description="Primary engine recommendation")
    confidence_score: float = Field(..., description="Classification confidence score")
    sub_components: Dict[str, bool] = Field(..., description="Sub-component breakdown")
    reasoning: str = Field(..., description="Reasoning for the classification")
    workflow_plan: Dict[str, Any] = Field(..., description="Workflow plan")
    user_request: str = Field(..., description="Original user request")


class RouteAndExecuteAck(BaseModel):
    """Acknowledgement returned after scheduling research execution"""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Execution status (accepted, running, completed, error)")
    classification: RouterResponse = Field(..., description="Classification details for the request")


@router.post("/classify", response_model=RouterResponse)
async def classify_request(request: RouterRequest):
    """Classify user request and recommend appropriate research engine"""
    try:
        app_state = get_app_state()
        smart_router = app_state["smart_router"]

        logger.info(f"Classifying request: {request.user_request[:100]}...")

        # Create classification request
        classification_request = ClassificationRequest(
            user_request=request.user_request,
            context=request.context
        )

        # Classify and route
        result = await smart_router.classify_and_route(classification_request)

        # Create response
        response = RouterResponse(
            primary_engine=result.primary_engine,
            confidence_score=result.confidence_score,
            sub_components=result.sub_components,
            reasoning=result.reasoning,
            workflow_plan=result.workflow_plan,
            user_request=request.user_request
        )

        logger.info(f"Classification completed: {result.primary_engine} (confidence: {result.confidence_score:.3f})")
        return response

    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.options("/route-and-execute")
async def route_and_execute_options():
    """Handle OPTIONS request for CORS preflight"""
    from fastapi.responses import Response

    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

@router.post("/route-and-execute", response_model=RouteAndExecuteAck)
async def route_and_execute(request: RouterRequest):
    """Classify the request and schedule research execution"""
    try:
        import uuid

        app_state = get_app_state()
        smart_router = app_state["smart_router"]
        engines = app_state["engines"]
        session_manager = app_state["session_manager"]
        base_llm_client = app_state["llm_client"]

        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        logger.info(f"Routing and scheduling request: {request.user_request[:100]}... (session={session_id})")

        existing_record = await session_manager.get_session(session_id)
        if existing_record and existing_record.get("status") in {"pending", "running"}:
            logger.info(
                "Session %s already active with status %s; skipping re-execution",
                session_id,
                existing_record.get("status"),
            )
            cached_classification = existing_record.get("classification") or {}
            try:
                classification_response = RouterResponse(**cached_classification)
            except Exception:
                classification_response = RouterResponse(
                    primary_engine=cached_classification.get("primary_engine", "SCIENTIFIC_RESEARCH"),
                    confidence_score=cached_classification.get("confidence_score", 0.0),
                    sub_components=cached_classification.get("sub_components", {}),
                    reasoning=cached_classification.get("reasoning", "Cached session"),
                    workflow_plan=cached_classification.get("workflow_plan", {}),
                    user_request=cached_classification.get("user_request", request.user_request),
                )

            return RouteAndExecuteAck(
                session_id=session_id,
                status=existing_record.get("status", "running"),
                classification=classification_response,
            )

        classification_request = ClassificationRequest(
            user_request=request.user_request,
            context=request.context
        )
        classification_result = await smart_router.classify_and_route(classification_request)

        classification_response = RouterResponse(
            primary_engine=classification_result.primary_engine,
            confidence_score=classification_result.confidence_score,
            sub_components=classification_result.sub_components,
            reasoning=classification_result.reasoning,
            workflow_plan=classification_result.workflow_plan,
            user_request=request.user_request
        )

        await progress_tracker.log_research_started(
            session_id=session_id,
            request=request.user_request,
            engine=classification_result.primary_engine
        )

        # Seed a minimal root node so late-joining clients can build a tree immediately
        try:
            await progress_tracker.log_research_progress(
                session_id=session_id,
                engine=classification_result.primary_engine.lower(),
                phase="session_initialized",
                progress=1.0,
                message="Session initialized",
                metadata={
                    "node_type": "engine",
                    "title": f"{classification_result.primary_engine.replace('_', ' ').title()} Engine",
                },
            )
        except Exception:
            # Non-fatal: if seeding fails, normal engine events will still build the tree
            pass

        await session_manager.create_session(
            session_id=session_id,
            request=request.user_request,
            classification=classification_response.dict()
        )

        engine_key = classification_result.primary_engine.lower().replace("_", "")

        async def execute_deep() -> Dict[str, Any]:
            engine = engines["deep"]
            streaming_llm_client = StreamingLLMClient(base_llm_client, session_id)
            original_llm_client = engine.llm_client
            engine.llm_client = streaming_llm_client

            original_search_clients = {}
            for search_name, search_engine in engine.search_engines.items():
                original_search_clients[search_name] = search_engine.llm_client
                search_engine.llm_client = streaming_llm_client

            try:
                execution_result = await engine.research(
                    request.user_request,
                    session_id=session_id
                )
            finally:
                engine.llm_client = original_llm_client
                for search_name, search_engine in engine.search_engines.items():
                    search_engine.llm_client = original_search_clients[search_name]

            await progress_tracker.log_research_completed(
                session_id=session_id,
                engine="deep_research",
                result_summary=execution_result.summary or "",
                metadata={
                    "sources_count": len(execution_result.sources),
                    "confidence_score": execution_result.confidence_score
                }
            )

            report_markdown = """# Deep Research Report\n\n"""
            report_markdown += f"## Summary\n{execution_result.summary}\n\n"
            if execution_result.key_findings:
                report_markdown += "## Key Findings\n" + "\n".join(
                    f"- {finding}" for finding in execution_result.key_findings
                ) + "\n\n"
            if execution_result.analysis:
                report_markdown += f"## Analysis\n{execution_result.analysis}\n\n"
            if execution_result.recommendations:
                report_markdown += "## Recommendations\n" + "\n".join(
                    f"- {rec}" for rec in execution_result.recommendations
                ) + "\n\n"

            return {
                "engine_used": "deep_research",
                "query": execution_result.query,
                "sources_count": len(execution_result.sources),
                "key_findings": execution_result.key_findings,
                "confidence_score": execution_result.confidence_score,
                "summary": execution_result.summary,
                "analysis": execution_result.analysis,
                "recommendations": execution_result.recommendations,
                "report_markdown": report_markdown
            }

        async def execute_code() -> Dict[str, Any]:
            engine = engines["code"]
            streaming_llm_client = StreamingLLMClient(base_llm_client, session_id)
            original_llm_client = engine.llm_client
            engine.llm_client = streaming_llm_client

            try:
                execution_result = await engine.research_code(
                    request.user_request,
                    session_id=session_id,
                    workflow_plan=classification_result.workflow_plan
                )
            finally:
                engine.llm_client = original_llm_client

            await progress_tracker.log_research_completed(
                session_id=session_id,
                engine="code_research",
                result_summary=(
                    f"Found {len(execution_result.repositories)} repositories "
                    f"with {len(execution_result.best_practices)} best practices"
                ),
                metadata={
                    "repositories_count": len(execution_result.repositories),
                    "confidence_score": execution_result.confidence_score,
                    "languages_found": list({repo.language for repo in execution_result.repositories})
                }
            )

            report_markdown = getattr(execution_result, "report_markdown", "") or """# Code Research Report\n\n"""
            if not getattr(execution_result, "report_markdown", None):
                report_markdown += f"## Summary\nIdentified {len(execution_result.repositories)} repositories with {len(execution_result.best_practices)} best practices.\n\n"
                if execution_result.repositories:
                    report_markdown += "## Repositories\n" + "\n".join(
                        f"- **{repo.name}** ({repo.language}) — {repo.description}" for repo in execution_result.repositories[:10]
                    ) + "\n\n"
                if execution_result.best_practices:
                    report_markdown += "## Best Practices\n" + "\n".join(
                        f"- {pattern.name}: {pattern.description}" for pattern in execution_result.best_practices[:10]
                    ) + "\n\n"
                if execution_result.integration_guide:
                    report_markdown += "## Integration Guide\n" + execution_result.integration_guide + "\n\n"
                if execution_result.recommendations:
                    report_markdown += "## Recommendations\n" + "\n".join(
                        f"- {rec}" for rec in execution_result.recommendations
                    ) + "\n\n"

            return {
                "engine_used": "code_research",
                "query": execution_result.query,
                "repositories_count": len(execution_result.repositories),
                "languages_found": list({repo.language for repo in execution_result.repositories}),
                "best_practices_count": len(execution_result.best_practices),
                "confidence_score": execution_result.confidence_score,
                "integration_guide": execution_result.integration_guide,
                "analysis": [analysis.summary for analysis in execution_result.analysis],
                "recommendations": execution_result.recommendations,
                "report_markdown": report_markdown,
                "repomaster_metadata": getattr(execution_result, "repomaster_metadata", None)
            }

        async def execute_scientific() -> Dict[str, Any]:
            engine = engines["scientific"]
            params = classification_result.workflow_plan
            streaming_llm_client = StreamingLLMClient(base_llm_client, session_id)
            original_llm_client = engine.llm_client
            engine.llm_client = streaming_llm_client

            # Enable OpenHands streaming if available
            if hasattr(engine, 'openhands_client') and engine.openhands_client:
                engine.openhands_client = enable_openhands_streaming(engine.openhands_client, session_id)

            try:
                execution_result = await engine.conduct_research(
                    request.user_request,
                    include_literature_review=params.get("include_literature_review", True),
                    include_code_analysis=params.get("include_code_analysis", True),
                    enable_iteration=params.get("enable_iteration", True),
                    session_id=session_id
                )
            finally:
                engine.llm_client = original_llm_client

            def _serialize_execution(exec_obj):
                return {
                    "execution_id": exec_obj.id,
                    "design_id": exec_obj.design_id,
                    "status": exec_obj.status.value,
                    "start_time": exec_obj.start_time.isoformat(),
                    "end_time": exec_obj.end_time.isoformat() if exec_obj.end_time else None,
                    "progress": exec_obj.progress,
                    "output_data": exec_obj.output_data,
                    "logs": exec_obj.logs,
                    "errors": exec_obj.errors,
                }

            def _serialize_result(res_obj):
                return {
                    "execution_id": res_obj.execution_id,
                    "hypothesis_id": res_obj.hypothesis_id,
                    "status": res_obj.status.value,
                    "data": res_obj.data,
                    "analysis": res_obj.analysis,
                    "statistical_tests": res_obj.statistical_tests,
                    "conclusions": res_obj.conclusions,
                    "confidence_score": res_obj.confidence_score,
                    "reproducibility_score": res_obj.reproducibility_score,
                    "limitations": res_obj.limitations,
                    "next_steps": res_obj.next_steps,
                }

            await progress_tracker.log_research_completed(
                session_id=session_id,
                engine="scientific_research",
                result_summary=(
                    f"Completed {execution_result.iteration_count} iterations "
                    f"with {len(execution_result.hypotheses)} hypotheses"
                ),
                metadata={
                    "hypotheses_count": len(execution_result.hypotheses),
                    "experiments_count": len(execution_result.experiments),
                    "iterations_completed": execution_result.iteration_count,
                    "confidence_score": execution_result.confidence_score,
                    "has_literature_review": execution_result.literature_review is not None,
                    "has_code_analysis": execution_result.code_analysis is not None
                }
            )

            summary_value = (
                execution_result.synthesis
                if isinstance(execution_result.synthesis, str)
                else str(execution_result.synthesis)
            )
            report_lines = ["# Scientific Research Report", ""]
            report_lines.append(f"## Research Question\n{execution_result.query}\n")

            if execution_result.hypotheses:
                report_lines.append("## Hypotheses")
                report_lines.extend(
                    f"- {hyp.statement}" for hyp in execution_result.hypotheses[:10]
                )
                report_lines.append("")

            if execution_result.experiments:
                report_lines.append("## Experiments")
                for exp in execution_result.experiments[:10]:
                    report_lines.append(f"- **{exp.name}** — {exp.description}")
                report_lines.append("")

            serialized_execs = [
                _serialize_execution(exec_obj)
                for exec_obj in execution_result.executions or []
            ]
            if serialized_execs:
                report_lines.append("## Execution Details")
                for exec_dict in serialized_execs:
                    workspace = exec_dict["output_data"].get("workspace_id") if exec_dict.get("output_data") else None
                    report_lines.append(
                        f"- Execution `{exec_dict['execution_id']}` ({exec_dict['status']})"
                    )
                    if workspace:
                        report_lines.append(f"  - Workspace: `{workspace}`")
                    if exec_dict.get("output_data", {}).get("generated_code"):
                        snippet = exec_dict["output_data"]["generated_code"][:200].replace("\n", " ")
                        report_lines.append(f"  - Generated code snippet: `{snippet}`…")
                    if exec_dict.get("errors"):
                        report_lines.append(f"  - Errors: {exec_dict['errors'][:1]}")
                report_lines.append("")

            serialized_results = [
                _serialize_result(res_obj)
                for res_obj in execution_result.results or []
            ]
            if serialized_results:
                report_lines.append("## Experiment Results")
                for res in serialized_results[:5]:
                    report_lines.append(
                        f"- Result `{res['execution_id']}` ({res['status']}) — Conclusions: { '; '.join(res['conclusions'] or []) or 'Pending' }"
                    )
                report_lines.append("")

            if isinstance(execution_result.synthesis, dict):
                report_lines.append("## Synthesis")
                for key, value in execution_result.synthesis.items():
                    if isinstance(value, list):
                        report_lines.append(f"- **{key.replace('_', ' ').title()}**:")
                        report_lines.extend(f"  - {item}" for item in value)
                    else:
                        report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                report_lines.append("")
            elif summary_value:
                report_lines.append(f"## Synthesis\n{summary_value}\n")

            if execution_result.recommendations:
                report_lines.append("## Recommendations")
                report_lines.extend(f"- {rec}" for rec in execution_result.recommendations)
                report_lines.append("")

            report_markdown = "\n".join(report_lines).strip() + "\n"

            # Determine completion status based on confidence threshold
            try:
                import os
                threshold = float(os.getenv("UAGENT_SCIENCE_COMPLETE_THRESHOLD", os.getenv("CONFIDENCE_THRESHOLD", "0.8")))
            except Exception:
                threshold = 0.8
            is_complete = (execution_result.confidence_score or 0.0) >= threshold

            return {
                "engine_used": "scientific_research",
                    "query": execution_result.query,
                    "hypotheses_count": len(execution_result.hypotheses),
                    "experiments_count": len(execution_result.experiments),
                    "iterations_completed": execution_result.iteration_count,
                    "confidence_score": execution_result.confidence_score,
                    "has_literature_review": execution_result.literature_review is not None,
                    "has_code_analysis": execution_result.code_analysis is not None,
                    "summary": summary_value,
                    "synthesis": execution_result.synthesis,
                    "recommendations": execution_result.recommendations,
                    "report_markdown": report_markdown,
                    "executions": serialized_execs,
                    "experiment_results": serialized_results,
                    "idea_evaluations": {
                        idea_id: asdict(eval_obj) if eval_obj else None
                        for idea_id, eval_obj in (execution_result.idea_evaluations or {}).items()
                    },
                "is_complete": bool(is_complete),
                "complete_criteria": {
                    "confidence_threshold": threshold,
                    "met": bool(is_complete),
                },
            }

        async def execute_research() -> None:
            await session_manager.set_status(session_id, "running")
            try:
                if engine_key == "deepresearch":
                    execution_payload = await execute_deep()
                elif engine_key == "coderesearch":
                    execution_payload = await execute_code()
                elif engine_key == "scientificresearch":
                    execution_payload = await execute_scientific()
                else:
                    raise ValueError(f"Unknown engine type: {classification_result.primary_engine}")

                result_payload = {
                    "classification": classification_response.dict(),
                    "execution": execution_payload
                }
                # Auto-continue option for incomplete scientific runs
                auto_passes = 0
                try:
                    import os
                    auto_passes = int(os.getenv("UAGENT_AUTO_CONTINUE_INCOMPLETE", "0"))
                except Exception:
                    auto_passes = 0

                if engine_key == "scientificresearch" and isinstance(execution_payload, dict):
                    passes = 0
                    while not execution_payload.get("is_complete", True) and passes < auto_passes:
                        passes += 1
                        # Re-run scientific engine once more
                        execution_payload = await execute_scientific()
                        result_payload["execution"] = execution_payload

                # Mark result completed only when complete or for non-scientific engines
                if engine_key != "scientificresearch" or execution_payload.get("is_complete", True):
                    await session_manager.set_result(session_id, result_payload)
                else:
                    await session_manager.set_status(session_id, "running")

            except Exception as exc:
                logger.error(
                    "Research execution failed for session %s: %s",
                    session_id,
                    exc,
                    exc_info=True
                )
                await session_manager.set_error(session_id, str(exc))
                await progress_tracker.log_error(
                    session_id=session_id,
                    engine=classification_result.primary_engine,
                    error=str(exc),
                    phase="execution"
                )

        task = asyncio.create_task(execute_research(), name=f"research_{session_id}")
        await session_manager.attach_task(session_id, task)

        return RouteAndExecuteAck(
            session_id=session_id,
            status="accepted",
            classification=classification_response
        )

    except Exception as e:
        logger.error(f"Route and execute scheduling failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route and execute failed: {str(e)}"
        )


@router.get("/engines")
async def list_available_engines():
    """List all available research engines and their capabilities"""
    return {
        "engines": {
            "DEEP_RESEARCH": {
                "name": "Deep Research Engine",
                "description": "Comprehensive multi-source research with web, academic, and technical sources",
                "capabilities": [
                    "Web search and analysis",
                    "Academic paper research",
                    "Technical documentation analysis",
                    "Multi-source synthesis",
                    "Key findings extraction"
                ],
                "best_for": [
                    "Market research",
                    "Literature reviews",
                    "Industry analysis",
                    "Competitive intelligence",
                    "General knowledge queries"
                ]
            },
            "CODE_RESEARCH": {
                "name": "Code Research Engine",
                "description": "Specialized research for code analysis, repositories, and development patterns",
                "capabilities": [
                    "GitHub repository analysis",
                    "Code pattern identification",
                    "Best practices extraction",
                    "Library and framework research",
                    "Integration guides generation"
                ],
                "best_for": [
                    "Technology evaluation",
                    "Library selection",
                    "Code architecture research",
                    "Development methodologies",
                    "Technical implementation guides"
                ]
            },
            "SCIENTIFIC_RESEARCH": {
                "name": "Scientific Research Engine",
                "description": "Advanced research engine with hypothesis generation, experimental design, and iterative refinement",
                "capabilities": [
                    "Hypothesis generation and testing",
                    "Experimental design",
                    "Literature review integration",
                    "Code analysis integration",
                    "Iterative refinement",
                    "Publication-ready output"
                ],
                "best_for": [
                    "Academic research",
                    "Scientific investigations",
                    "Experimental validation",
                    "Research proposals",
                    "Complex multi-faceted studies"
                ]
            }
        },
        "routing_logic": {
            "description": "Requests are automatically classified using AI to determine the most appropriate engine",
            "factors": [
                "Query complexity and scope",
                "Research methodology requirements",
                "Output format expectations",
                "Domain expertise needed",
                "Integration requirements"
            ]
        }
    }


@router.get("/status")
async def router_status():
    """Get smart router system status"""
    try:
        app_state = get_app_state()
        smart_router = app_state["smart_router"]
        engines = app_state["engines"]

        # Check engine availability
        engine_status = {}
        for name, engine in engines.items():
            engine_status[name] = {
                "available": engine is not None,
                "type": engine.__class__.__name__ if engine else None
            }

        return {
            "status": "operational",
            "smart_router": {
                "available": smart_router is not None,
                "type": smart_router.__class__.__name__ if smart_router else None
            },
            "engines": engine_status,
            "total_engines": len(engines),
            "cache_available": app_state.get("cache") is not None,
            "llm_client_available": app_state.get("llm_client") is not None
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "smart_router": {"available": False},
            "engines": {},
            "total_engines": 0
        }
