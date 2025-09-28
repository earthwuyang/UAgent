import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import {
  Clock,
  RefreshCw,
  Play,
  FolderOpen,
  Calendar
} from 'lucide-react';
import { clsx } from 'clsx';

interface ActiveExperiment {
  session_id: string;
  original_query: string;
  status: string;
  start_time: string;
  workspace_path: string | null;
}

interface ActiveExperimentsResponse {
  active_experiments: ActiveExperiment[];
  count: number;
}

const ActiveExperiments: React.FC = () => {
  const [activeExperiments, setActiveExperiments] = useState<ActiveExperiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  const [consecutiveErrors, setConsecutiveErrors] = useState(0);

  const fetchActiveExperiments = async () => {
    setLoading(true);
    setError(null);
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';

      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch(`${baseUrl}/api/experiments/active`, {
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
        },
        // Bypass proxy for localhost
        ...(baseUrl.includes('localhost') && {
          mode: 'cors' as RequestMode,
        }),
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: ActiveExperimentsResponse = await response.json();
      setActiveExperiments(data.active_experiments);
      setConsecutiveErrors(0); // Reset error count on success
      setAutoRefreshEnabled(true); // Re-enable auto refresh on success
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch active experiments';
      setError(errorMessage);

      // Increment consecutive errors
      const newErrorCount = consecutiveErrors + 1;
      setConsecutiveErrors(newErrorCount);

      // Disable auto-refresh after 3 consecutive errors
      if (newErrorCount >= 3) {
        setAutoRefreshEnabled(false);
        setError(`${errorMessage} (Auto-refresh disabled after ${newErrorCount} failures)`);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchActiveExperiments();
  }, []);

  useEffect(() => {
    if (!autoRefreshEnabled) return;

    // Auto-refresh every 30 seconds if enabled
    const interval = setInterval(() => {
      if (autoRefreshEnabled) {
        fetchActiveExperiments();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefreshEnabled, consecutiveErrors]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getRunningTime = (startTime: string) => {
    const start = new Date(startTime);
    const now = new Date();
    const diffMs = now.getTime() - start.getTime();
    const diffMinutes = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMinutes / 60);

    if (diffHours > 0) {
      return `${diffHours}h ${diffMinutes % 60}m`;
    } else {
      return `${diffMinutes}m`;
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Play className="h-5 w-5 text-blue-500" />
              <span>Active Experiments</span>
              <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                {activeExperiments.length}
              </Badge>
              {autoRefreshEnabled && (
                <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
                  Auto-refresh
                </Badge>
              )}
            </CardTitle>
            <CardDescription>Currently running research experiments</CardDescription>
          </div>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setConsecutiveErrors(0);
                setAutoRefreshEnabled(true);
                fetchActiveExperiments();
              }}
              disabled={loading}
            >
              <RefreshCw className={clsx('h-4 w-4 mr-2', loading && 'animate-spin')} />
              {autoRefreshEnabled ? 'Refresh' : 'Retry'}
            </Button>
            {!autoRefreshEnabled && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  setConsecutiveErrors(0);
                  setAutoRefreshEnabled(true);
                  setError(null);
                }}
              >
                Enable Auto-refresh
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert className="border-red-200 bg-red-50 mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {loading && activeExperiments.length === 0 ? (
          <div className="text-center py-8">
            <RefreshCw className="h-8 w-8 animate-spin mx-auto text-gray-400" />
            <p className="text-gray-500 mt-2">Loading active experiments...</p>
          </div>
        ) : activeExperiments.length === 0 ? (
          <div className="text-center py-8">
            <Play className="h-8 w-8 mx-auto text-gray-400" />
            <p className="text-gray-500 mt-2">No active experiments</p>
            <p className="text-sm text-gray-400">All experiments have completed</p>
          </div>
        ) : (
          <div className="space-y-4">
            {activeExperiments.map((experiment) => (
              <Card key={experiment.session_id} className="border-l-4 border-l-blue-500">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="text-lg font-semibold text-gray-900 truncate">
                          {experiment.session_id}
                        </h3>
                        <Badge variant="default" className="bg-blue-100 text-blue-800 border-blue-300">
                          Running
                        </Badge>
                      </div>

                      <p className="text-gray-600 mb-3 line-clamp-2">
                        {experiment.original_query}
                      </p>

                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-3 w-3" />
                          <span>{formatDate(experiment.start_time)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Clock className="h-3 w-3" />
                          <span>Running for {getRunningTime(experiment.start_time)}</span>
                        </div>
                        {experiment.workspace_path && (
                          <div className="flex items-center space-x-1">
                            <FolderOpen className="h-3 w-3" />
                            <span className="truncate max-w-xs">{experiment.workspace_path}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ActiveExperiments;