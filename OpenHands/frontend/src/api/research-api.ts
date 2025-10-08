import { openHands } from '#/api/open-hands-axios';
import { AxiosResponse } from 'axios';

// Type definitions
export type ExperimentStatus = 'idle' | 'running' | 'paused' | 'complete' | 'failed' | 'cancelled';

export type ExperimentControlAction = 'start' | 'pause' | 'resume' | 'cancel' | 'cancel_node';

export interface ExperimentControlRequest {
  action: ExperimentControlAction;
  target?: Record<string, string>;
  payload?: any;
}

export interface ExperimentControlResponse {
  experiment_id: string;
  action: string;
  status: string;
  message: string;
  target?: any;
  payload?: any;
}

export interface ExperimentStatusResponse {
  experiment_id: string;
  status: ExperimentStatus;
  stats: Record<string, any>;
  adapters: Record<string, Record<string, any>>;
  active_branches: Array<Record<string, any>>;
}

export interface ExperimentProgressInfo {
  total_nodes: number;
  completed: number;
  running: number;
  failed: number;
}

export interface AdapterStatus {
  status: string;
  current_action?: string;
}

export interface ActiveBranch {
  branch_id: string;
  adapter: string;
  elapsed_time: number;
}

// Custom error class for API errors
export class ResearchAPIError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'ResearchAPIError';
  }
}

// API methods
export async function startExperiment(experimentId: string, goal: string): Promise<ExperimentControlResponse> {
  try {
    const response: AxiosResponse<ExperimentControlResponse> = await openHands.patch(
      `/api/research/experiments/${experimentId}`,
      {
        action: 'start',
        payload: { goal }
      }
    );
    // Return the control response instead of trying to fetch status
    // Let callers poll status separately to avoid 404 issues
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || 'Failed to start experiment';
    throw new ResearchAPIError(message, error.response?.status);
  }
}

export async function getExperimentStatus(experimentId: string): Promise<ExperimentStatusResponse> {
  try {
    const response: AxiosResponse<ExperimentStatusResponse> = await openHands.get(
      `/api/research/experiments/${experimentId}/status`
    );
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || 'Failed to fetch experiment status';
    throw new ResearchAPIError(message, error.response?.status);
  }
}

export async function controlExperiment(
  experimentId: string,
  action: ExperimentControlAction,
  payload?: any
): Promise<ExperimentControlResponse> {
  try {
    const response: AxiosResponse<ExperimentControlResponse> = await openHands.patch(
      `/api/research/experiments/${experimentId}`,
      {
        action,
        payload
      }
    );
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || `Failed to ${action} experiment`;
    throw new ResearchAPIError(message, error.response?.status);
  }
}

export async function cancelNode(experimentId: string, nodeId: string): Promise<ExperimentControlResponse> {
  try {
    const response: AxiosResponse<ExperimentControlResponse> = await openHands.patch(
      `/api/research/experiments/${experimentId}`,
      {
        action: 'cancel_node',
        target: { node_id: nodeId }
      }
    );
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || 'Failed to cancel node';
    throw new ResearchAPIError(message, error.response?.status);
  }
}

export async function getExperimentEvents(
  experimentId: string,
  sinceVersion?: number,
  limit?: number
): Promise<{
  experiment_id: string;
  since_version: number;
  current_version: number;
  events: any[];
  has_more: boolean;
}> {
  try {
    const params = new URLSearchParams();
    if (sinceVersion !== undefined) params.append('since_version', sinceVersion.toString());
    if (limit !== undefined) params.append('limit', limit.toString());
    
    const response: AxiosResponse<{
      experiment_id: string;
      since_version: number;
      current_version: number;
      events: any[];
      has_more: boolean;
    }> = await openHands.get(
      `/api/research/experiments/${experimentId}/events?${params.toString()}`
    );
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || 'Failed to fetch experiment events';
    throw new ResearchAPIError(message, error.response?.status);
  }
}