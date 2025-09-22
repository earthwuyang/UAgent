// API Service for UAgent Frontend

import axios from 'axios';
import type {
  ClassificationRequest,
  ClassificationResult,
  RouteAndExecuteAck,
  RouteAndExecuteResponse,
  EnginesResponse,
  SystemStatus,
  HealthResponse,
  DeepResearchRequest,
  CodeResearchRequest,
  ScientificResearchRequest,
  DeepResearchResponse,
  CodeResearchResponse,
  ScientificResearchResponse,
  ResearchSession,
  SessionHistoryResponse,
  SessionListingResponse,
} from '../types/api';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 minutes for research tasks
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    throw error;
  }
);

export class UAgentAPI {
  // Health and Status
  static async getHealth(): Promise<HealthResponse> {
    const response = await axios.get('/health'); // Use root health endpoint
    return response.data;
  }

  static async getSystemStatus(): Promise<SystemStatus> {
    const response = await api.get('/router/status');
    return response.data;
  }

  static async getEngineInfo(): Promise<EnginesResponse> {
    const response = await api.get('/router/engines');
    return response.data;
  }

  // Smart Routing
  static async classifyRequest(request: ClassificationRequest): Promise<ClassificationResult> {
    const response = await api.post('/router/classify', request);
    return response.data;
  }

  static async routeAndExecute(request: ClassificationRequest): Promise<RouteAndExecuteAck> {
    const response = await api.post('/router/route-and-execute', request);
    return response.data;
  }

  // Research Engines
  static async conductDeepResearch(request: DeepResearchRequest): Promise<DeepResearchResponse> {
    const response = await api.post('/research/deep', request);
    return response.data;
  }

  static async conductCodeResearch(request: CodeResearchRequest): Promise<CodeResearchResponse> {
    const response = await api.post('/research/code', request);
    return response.data;
  }

  static async conductScientificResearch(request: ScientificResearchRequest): Promise<ScientificResearchResponse> {
    const response = await api.post('/research/scientific', request);
    return response.data;
  }

  // Research Session Management
  static async getResearchSessions(): Promise<SessionListingResponse> {
    const response = await api.get('/research/sessions');
    return response.data;
  }

  static async getResearchSession(researchId: string): Promise<any> {
    const response = await api.get(`/research/sessions/${researchId}`);
    return response.data;
  }

  static async getFullResearchResult(researchId: string): Promise<RouteAndExecuteResponse | null> {
    const response = await api.get(`/research/sessions/${researchId}/full`);
    const data = response.data;
    if (!data || !data.result) {
      return null;
    }
    return data.result as RouteAndExecuteResponse;
  }

  static async getSessionReport(researchId: string): Promise<RouteAndExecuteResponse | null> {
    return this.getFullResearchResult(researchId);
  }

  static async deleteResearchSession(researchId: string): Promise<{ message: string }> {
    const response = await api.delete(`/research/sessions/${researchId}`);
    return response.data;
  }

  static async getSessionHistory(researchId: string): Promise<SessionHistoryResponse> {
    const response = await api.get(`/research/sessions/${researchId}/history`);
    return response.data;
  }
}

export default UAgentAPI;
