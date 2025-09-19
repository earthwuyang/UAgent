/**
 * ResearchDashboard Component
 * Comprehensive real-time research visualization dashboard
 * Integrates all research engines with live progress tracking
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';
import { ResearchProgressStream } from './ResearchProgressStream';
import ResearchTreeVisualization from './ResearchTreeVisualization';

interface ResearchRequest {
  user_request: string;
  session_id?: string;
  context?: any;
}

interface ResearchResponse {
  classification: {
    primary_engine: string;
    confidence_score: number;
    reasoning: string;
  };
  execution: any;
  session_id?: string;
}

export const ResearchDashboard: React.FC = () => {
  const [request, setRequest] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [progressEvents, setProgressEvents] = useState<any[]>([]);

  const exampleRequests = [
    {
      title: "Deep Research",
      text: "Research the latest developments in artificial intelligence and machine learning for 2024, including breakthrough technologies and industry trends",
      engine: "DEEP_RESEARCH"
    },
    {
      title: "Code Research",
      text: "Analyze the code quality and architecture patterns in popular Python FastAPI projects on GitHub",
      engine: "CODE_RESEARCH"
    },
    {
      title: "Scientific Research",
      text: "Design and conduct experiments to test the effectiveness of different transformer attention mechanisms on natural language understanding tasks",
      engine: "SCIENTIFIC_RESEARCH"
    }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!request.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults(null);

    // Generate new session ID for this research
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);

    try {
      const requestData: ResearchRequest = {
        user_request: request,
        session_id: newSessionId,
        context: {
          timestamp: new Date().toISOString(),
          dashboard_session: true
        }
      };

      const response = await fetch('http://localhost:8012/api/router/route-and-execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);

    } catch (err) {
      console.error('Research request failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (exampleText: string) => {
    setRequest(exampleText);
  };

  const getEngineColor = (engine: string) => {
    switch (engine) {
      case 'DEEP_RESEARCH': return 'bg-blue-500';
      case 'CODE_RESEARCH': return 'bg-green-500';
      case 'SCIENTIFIC_RESEARCH': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };

  const getEngineIcon = (engine: string) => {
    switch (engine) {
      case 'DEEP_RESEARCH': return '🔍';
      case 'CODE_RESEARCH': return '💻';
      case 'SCIENTIFIC_RESEARCH': return '🧪';
      default: return '📋';
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">🤖 Universal Agent Research Dashboard</h1>
        <p className="text-muted-foreground">
          Intelligent multi-engine research with real-time progress visualization
        </p>
        {connected && (
          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
            🟢 Live Connection Active
          </Badge>
        )}
      </div>

      {/* Research Input */}
      <Card>
        <CardHeader>
          <CardTitle>Start New Research</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Textarea
                placeholder="Describe your research goal in natural language..."
                value={request}
                onChange={(e) => setRequest(e.target.value)}
                className="min-h-[100px]"
              />
            </div>

            <div className="flex justify-between items-center">
              <Button type="submit" disabled={isLoading || !request.trim()}>
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    🚀 Start Research
                  </>
                )}
              </Button>

              {results && (
                <Badge className={`${getEngineColor(results.classification.primary_engine)} text-white`}>
                  {getEngineIcon(results.classification.primary_engine)} {results.classification.primary_engine}
                  <span className="ml-1">({Math.round(results.classification.confidence_score * 100)}%)</span>
                </Badge>
              )}
            </div>
          </form>

          {/* Example Requests */}
          <div className="mt-6">
            <h3 className="text-sm font-medium mb-3">Example Research Requests:</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {exampleRequests.map((example, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => handleExampleClick(example.text)}
                  className="text-left h-auto p-3 flex flex-col items-start space-y-1"
                >
                  <div className="flex items-center gap-2">
                    <span>{getEngineIcon(example.engine)}</span>
                    <span className="font-medium text-xs">{example.title}</span>
                  </div>
                  <span className="text-xs text-muted-foreground line-clamp-2">
                    {example.text.slice(0, 80)}...
                  </span>
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>
            ❌ {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Results and Progress */}
      {(sessionId || results) && (
        <Tabs defaultValue="progress" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="progress">🔄 Live Progress</TabsTrigger>
            <TabsTrigger value="tree">🌳 Tree View</TabsTrigger>
            <TabsTrigger value="results">📊 Results</TabsTrigger>
          </TabsList>

          <TabsContent value="progress" className="space-y-4">
            {sessionId && (
              <ResearchProgressStream
                sessionId={sessionId}
                onConnectionChange={setConnected}
                onEventsUpdate={setProgressEvents}
              />
            )}
          </TabsContent>

          <TabsContent value="tree" className="space-y-4">
            {sessionId && (
              <ResearchTreeVisualization
                sessionId={sessionId}
                events={progressEvents}
              />
            )}
          </TabsContent>

          <TabsContent value="results" className="space-y-4">
            {results ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Classification Results */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      🎯 Classification Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <span className="font-medium">Selected Engine:</span>
                        <Badge className={`${getEngineColor(results.classification.primary_engine)} text-white`}>
                          {getEngineIcon(results.classification.primary_engine)} {results.classification.primary_engine}
                        </Badge>
                      </div>
                    </div>

                    <div>
                      <span className="font-medium">Confidence Score:</span>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${results.classification.confidence_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm">
                          {Math.round(results.classification.confidence_score * 100)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <span className="font-medium">Reasoning:</span>
                      <p className="text-sm text-muted-foreground mt-1">
                        {results.classification.reasoning}
                      </p>
                    </div>
                  </CardContent>
                </Card>

                {/* Execution Results */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      ⚡ Execution Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {results.execution && (
                        <div className="p-4 bg-muted rounded-lg">
                          <h4 className="font-medium mb-2">Engine Output:</h4>
                          <pre className="text-xs overflow-auto max-h-40">
                            {JSON.stringify(results.execution, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="text-center py-8">
                  <div className="text-4xl mb-2">⏳</div>
                  <p className="text-muted-foreground">
                    Research results will appear here once processing is complete
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      )}

      {/* Instructions */}
      {!sessionId && !results && (
        <Card>
          <CardContent className="text-center py-8 space-y-4">
            <div className="text-6xl">🔬</div>
            <div>
              <h3 className="text-lg font-medium mb-2">Welcome to Universal Agent</h3>
              <p className="text-muted-foreground">
                Enter your research request above to get started. The system will automatically:
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              <div className="space-y-2">
                <div className="text-2xl">🎯</div>
                <h4 className="font-medium">Smart Routing</h4>
                <p className="text-sm text-muted-foreground">
                  Automatically classify your request and route to the best research engine
                </p>
              </div>
              <div className="space-y-2">
                <div className="text-2xl">🔄</div>
                <h4 className="font-medium">Live Progress</h4>
                <p className="text-sm text-muted-foreground">
                  Watch real-time progress with live updates from all research engines
                </p>
              </div>
              <div className="space-y-2">
                <div className="text-2xl">📊</div>
                <h4 className="font-medium">Rich Results</h4>
                <p className="text-sm text-muted-foreground">
                  Get comprehensive results with multi-engine coordination
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};