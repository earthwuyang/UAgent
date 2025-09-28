import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { ScrollArea } from '../ui/scroll-area';
import { Alert, AlertDescription } from '../ui/alert';
import {
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Calendar,
  FileText,
  BarChart3,
  RefreshCw,
  FolderOpen,
  Search
} from 'lucide-react';
import { clsx } from 'clsx';
import ActiveExperiments from './ActiveExperiments';

interface ExperimentData {
  session_id: string;
  original_query: string;
  status: string;
  start_time: string;
  end_time: string | null;
  duration_seconds: number | null;
  readable_name: string | null;
  success: boolean;
  arxiv_path: string;
  final_result: any;
  error_message: string | null;
  archived_at: string;
}

interface ExperimentStats {
  active_count: number;
  archived_count: number;
  successful_count: number;
  failed_count: number;
  interrupted_count: number;
  arxiv_path: string | null;
}

interface ExperimentListResponse {
  experiments: ExperimentData[];
  total_count: number;
  successful_count: number;
  failed_count: number;
  interrupted_count: number;
}

const ExperimentHistory: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [stats, setStats] = useState<ExperimentStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<string>('all');
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentData | null>(null);

  const fetchExperiments = async (statusFilter?: string) => {
    setLoading(true);
    setError(null);
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
      const url = statusFilter
        ? `${baseUrl}/api/experiments/archived?status_filter=${statusFilter}&limit=100`
        : `${baseUrl}/api/experiments/archived?limit=100`;

      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

      const response = await fetch(url, {
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

      const data: ExperimentListResponse = await response.json();
      setExperiments(data.experiments);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch experiments');
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';

      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch(`${baseUrl}/api/experiments/stats`, {
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

      const data: ExperimentStats = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch experiment stats:', err);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchExperiments();
  }, []);

  useEffect(() => {
    if (selectedTab === 'all') {
      fetchExperiments();
    } else {
      fetchExperiments(selectedTab);
    }
  }, [selectedTab]);

  const getStatusIcon = (experiment: ExperimentData) => {
    if (experiment.success) {
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    } else if (experiment.status === 'interrupted') {
      return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    } else {
      return <XCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getStatusBadge = (experiment: ExperimentData) => {
    if (experiment.success) {
      return <Badge variant="default" className="bg-green-100 text-green-800 border-green-300">Success</Badge>;
    } else if (experiment.status === 'interrupted') {
      return <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 border-yellow-300">Interrupted</Badge>;
    } else {
      return <Badge variant="destructive" className="bg-red-100 text-red-800 border-red-300">Failed</Badge>;
    }
  };

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return 'Unknown';
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${Math.floor(seconds % 60)}s`;
    } else {
      return `${Math.floor(seconds)}s`;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Experiment History</h1>
          <p className="text-gray-600 mt-1">View and manage your research experiment archive</p>
        </div>
        <Button onClick={() => fetchExperiments(selectedTab === 'all' ? undefined : selectedTab)} disabled={loading}>
          <RefreshCw className={clsx('h-4 w-4 mr-2', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-blue-500" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Active</p>
                  <p className="text-2xl font-bold text-blue-600">{stats.active_count}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Successful</p>
                  <p className="text-2xl font-bold text-green-600">{stats.successful_count}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <XCircle className="h-4 w-4 text-red-500" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Failed</p>
                  <p className="text-2xl font-bold text-red-600">{stats.failed_count}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <AlertCircle className="h-4 w-4 text-yellow-500" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Interrupted</p>
                  <p className="text-2xl font-bold text-yellow-600">{stats.interrupted_count}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <BarChart3 className="h-4 w-4 text-purple-500" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Total</p>
                  <p className="text-2xl font-bold text-purple-600">{stats.archived_count}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Active Experiments */}
      <ActiveExperiments />

      {/* Experiment List */}
      <Card>
        <CardHeader>
          <CardTitle>Experiment Archive</CardTitle>
          <CardDescription>Browse your experiment history by status</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={selectedTab} onValueChange={setSelectedTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="all">All ({experiments.length})</TabsTrigger>
              <TabsTrigger value="successful">
                Successful ({stats?.successful_count || 0})
              </TabsTrigger>
              <TabsTrigger value="failed">
                Failed ({stats?.failed_count || 0})
              </TabsTrigger>
              <TabsTrigger value="interrupted">
                Interrupted ({stats?.interrupted_count || 0})
              </TabsTrigger>
            </TabsList>

            <TabsContent value={selectedTab} className="mt-6">
              <div className="grid gap-4">
                {loading ? (
                  <div className="text-center py-8">
                    <RefreshCw className="h-8 w-8 animate-spin mx-auto text-gray-400" />
                    <p className="text-gray-500 mt-2">Loading experiments...</p>
                  </div>
                ) : experiments.length === 0 ? (
                  <div className="text-center py-8">
                    <Search className="h-8 w-8 mx-auto text-gray-400" />
                    <p className="text-gray-500 mt-2">No experiments found</p>
                  </div>
                ) : (
                  <div className="grid gap-4">
                    {experiments.map((experiment) => (
                      <Card
                        key={experiment.session_id}
                        className={clsx(
                          'cursor-pointer transition-all hover:shadow-md',
                          selectedExperiment?.session_id === experiment.session_id && 'ring-2 ring-blue-500'
                        )}
                        onClick={() => setSelectedExperiment(
                          selectedExperiment?.session_id === experiment.session_id ? null : experiment
                        )}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-2 mb-2">
                                {getStatusIcon(experiment)}
                                <h3 className="text-lg font-semibold text-gray-900 truncate">
                                  {experiment.readable_name || experiment.session_id}
                                </h3>
                                {getStatusBadge(experiment)}
                              </div>

                              <p className="text-gray-600 mb-3 line-clamp-2">
                                {experiment.original_query}
                              </p>

                              <div className="flex items-center space-x-4 text-sm text-gray-500">
                                <div className="flex items-center space-x-1">
                                  <Calendar className="h-3 w-3" />
                                  <span>{formatDate(experiment.start_time)}</span>
                                </div>
                                {experiment.duration_seconds && (
                                  <div className="flex items-center space-x-1">
                                    <Clock className="h-3 w-3" />
                                    <span>{formatDuration(experiment.duration_seconds)}</span>
                                  </div>
                                )}
                                <div className="flex items-center space-x-1">
                                  <FolderOpen className="h-3 w-3" />
                                  <span className="truncate max-w-xs">{experiment.arxiv_path}</span>
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Expanded Details */}
                          {selectedExperiment?.session_id === experiment.session_id && (
                            <div className="mt-4 pt-4 border-t border-gray-200">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-2">Details</h4>
                                  <div className="space-y-1 text-sm">
                                    <div><span className="font-medium">Session ID:</span> {experiment.session_id}</div>
                                    <div><span className="font-medium">Started:</span> {formatDate(experiment.start_time)}</div>
                                    {experiment.end_time && (
                                      <div><span className="font-medium">Ended:</span> {formatDate(experiment.end_time)}</div>
                                    )}
                                    <div><span className="font-medium">Archived:</span> {formatDate(experiment.archived_at)}</div>
                                    {experiment.duration_seconds && (
                                      <div><span className="font-medium">Duration:</span> {formatDuration(experiment.duration_seconds)}</div>
                                    )}
                                  </div>
                                </div>

                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-2">Results</h4>
                                  {experiment.final_result ? (
                                    <ScrollArea className="h-32 w-full rounded border bg-gray-50 p-2">
                                      <pre className="text-xs">
                                        {JSON.stringify(experiment.final_result, null, 2)}
                                      </pre>
                                    </ScrollArea>
                                  ) : experiment.error_message ? (
                                    <div className="p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                                      {experiment.error_message}
                                    </div>
                                  ) : (
                                    <p className="text-sm text-gray-500">No result data available</p>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default ExperimentHistory;