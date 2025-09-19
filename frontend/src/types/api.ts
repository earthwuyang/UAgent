// API Types for UAgent Frontend

export interface ClassificationRequest {
  user_request: string;
  user_id?: string;
  session_id?: string;
  context?: Record<string, any>;
}

export interface ClassificationResult {
  primary_engine: string;
  confidence_score: number;
  sub_components: Record<string, boolean>;
  reasoning: string;
  workflow_plan: WorkflowPlan;
  user_request: string;
}

export interface WorkflowPlan {
  primary_engine: string;
  complexity_level: number;
  sub_workflows: SubWorkflow[];
  iteration_enabled: boolean;
  feedback_loops: boolean;
  max_iterations?: number;
}

export interface SubWorkflow {
  engine: string;
  phase: string;
  priority: string;
  description: string;
  includes_openhands?: boolean;
}

export interface ResearchRequest {
  query: string;
  user_id?: string;
  session_id?: string;
}

export interface DeepResearchRequest extends ResearchRequest {
  sources?: string[];
  max_sources_per_type?: number;
}

export interface CodeResearchRequest extends ResearchRequest {
  language?: string;
  include_analysis?: boolean;
  max_repositories?: number;
}

export interface ScientificResearchRequest extends ResearchRequest {
  include_literature_review?: boolean;
  include_code_analysis?: boolean;
  enable_iteration?: boolean;
  max_iterations?: number;
  confidence_threshold?: number;
}

export interface ResearchResponse {
  research_id: string;
  status: string;
  query: string;
}

export interface DeepResearchResponse extends ResearchResponse {
  sources_count: number;
  key_findings: string[];
  confidence_score: number;
  analysis_summary: string;
  recommendations: string[];
}

export interface CodeResearchResponse extends ResearchResponse {
  repositories_count: number;
  languages_found: string[];
  best_practices_count: number;
  confidence_score: number;
  integration_guide_length: number;
}

export interface ScientificResearchResponse extends ResearchResponse {
  hypotheses_count: number;
  experiments_count: number;
  iterations_completed: number;
  confidence_score: number;
  has_literature_review: boolean;
  has_code_analysis: boolean;
  publication_draft_length: number;
}

export interface RouteAndExecuteResponse {
  classification: {
    primary_engine: string;
    confidence_score: number;
    reasoning: string;
  };
  execution: {
    engine_used: string;
    query: string;
    [key: string]: any; // Dynamic fields based on engine type
    report_markdown?: string;
    summary?: string;
    analysis?: string | string[];
    recommendations?: string[];
  };
}

export interface RouteAndExecuteAck {
  session_id: string;
  status: string;
  classification: ClassificationResult;
}

export interface EngineInfo {
  name: string;
  description: string;
  capabilities: string[];
  best_for: string[];
}

export interface EnginesResponse {
  engines: Record<string, EngineInfo>;
  routing_logic: {
    description: string;
    factors: string[];
  };
}

export interface SystemStatus {
  status: string;
  smart_router: {
    available: boolean;
    type?: string;
  };
  engines: Record<string, {
    available: boolean;
    type?: string;
  }>;
  total_engines: number;
  cache_available: boolean;
  llm_client_available: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  engines: Record<string, string>;
  smart_router: string;
}

export type EngineType = 'DEEP_RESEARCH' | 'CODE_RESEARCH' | 'SCIENTIFIC_RESEARCH';

export interface ResearchSession {
  session_id?: string;
  research_id?: string;
  type?: string;
  query: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  has_result?: boolean;
  error?: string | null;
}
