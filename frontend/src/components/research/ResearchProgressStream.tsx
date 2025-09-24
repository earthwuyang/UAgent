/**
 * ResearchProgressStream Component
 * Real-time research progress visualization with live WebSocket updates
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { ScrollArea } from '../ui/scroll-area';
import { Separator } from '../ui/separator';
import { WS_BASE_URL } from '../../config';
import UAgentAPI from '../../services/api';
import type {
  ProgressEvent,
  JournalEntry,
  EngineStatus,
  SessionHistoryResponse,
} from '../../types/api';

interface ResearchProgressStreamProps {
  sessionId: string;
  onConnectionChange?: (connected: boolean) => void;
  onEventsUpdate?: (events: ProgressEvent[]) => void;
}

export const ResearchProgressStream: React.FC<ResearchProgressStreamProps> = ({
  sessionId,
  onConnectionChange,
  onEventsUpdate
}) => {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);
  const [engineStatuses, setEngineStatuses] = useState<Record<string, EngineStatus>>({});
  const [connected, setConnected] = useState(false);
  const [overallProgress, setOverallProgress] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const engineWsRef = useRef<WebSocket | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);
  const scrollTimeoutRef = useRef<number | null>(null);
  const eventSignaturesRef = useRef<Set<string>>(new Set());
  const journalSignaturesRef = useRef<Set<string>>(new Set());

  const scrollToBottom = () => {
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    scrollTimeoutRef.current = window.setTimeout(() => {
      eventsEndRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' });
    }, 16);
  };

  const makeEventSignature = (evt: ProgressEvent): string => (
    `${evt.event_type}|${evt.timestamp}|${evt.source}|${evt.message ?? ''}`
  );

  const makeJournalSignature = (entry: JournalEntry): string => (
    `${entry.timestamp}|${entry.engine}|${entry.phase}|${entry.message}`
  );

  const pushEvent = (evt: ProgressEvent) => {
    const signature = makeEventSignature(evt);
    if (eventSignaturesRef.current.has(signature)) {
      return;
    }
    eventSignaturesRef.current.add(signature);
    setEvents((prev) => {
      const next = [...prev, evt];
      return next;
    });
    if (typeof evt.progress_percentage === 'number') {
      setOverallProgress(evt.progress_percentage);
    }
  };

  const pushJournalEntry = (entry: JournalEntry) => {
    const signature = makeJournalSignature(entry);
    if (journalSignaturesRef.current.has(signature)) {
      return;
    }
    journalSignaturesRef.current.add(signature);
    setJournalEntries((prev) => [...prev, entry]);
  };

  const applySnapshot = (snapshot: SessionHistoryResponse) => {
    const snapshotEvents = snapshot.events ?? [];
    const snapshotJournal = snapshot.journal_entries ?? [];

    eventSignaturesRef.current.clear();
    journalSignaturesRef.current.clear();

    snapshotEvents.forEach((evt) => {
      eventSignaturesRef.current.add(makeEventSignature(evt));
    });
    snapshotJournal.forEach((entry) => {
      journalSignaturesRef.current.add(makeJournalSignature(entry));
    });

    setEvents(snapshotEvents);
    setJournalEntries(snapshotJournal);
    setEngineStatuses(snapshot.engine_statuses ?? {});

    const latestWithProgress = [...snapshotEvents]
      .reverse()
      .find((evt) => typeof evt.progress_percentage === 'number');

    if (latestWithProgress && typeof latestWithProgress.progress_percentage === 'number') {
      setOverallProgress(latestWithProgress.progress_percentage);
    } else if (snapshotEvents.length === 0) {
      setOverallProgress(0);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [events, journalEntries]);

  useEffect(() => {
    setEvents([]);
    setJournalEntries([]);
    setEngineStatuses({});
    setOverallProgress(0);
    setConnected(false);
    eventSignaturesRef.current.clear();
    journalSignaturesRef.current.clear();
  }, [sessionId]);

  // Cancel any pending scroll timers on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Propagate events to parent after render to avoid render-phase updates
  useEffect(() => {
    if (onEventsUpdate) {
      onEventsUpdate(events);
    }
  }, [events, onEventsUpdate]);

  useEffect(() => {
    if (!sessionId) return;

    let isCleanup = false;
    let progressReconnectTimeout: number | null = null;
    let engineReconnectTimeout: number | null = null;

    const loadSnapshot = async () => {
      try {
        const snapshot = await UAgentAPI.getSessionHistory(sessionId);
        if (!isCleanup) {
          applySnapshot(snapshot);
        }
      } catch (error) {
        console.error('Failed to load session snapshot:', error);
      }
    };

    // Connect to research progress WebSocket
    const connectProgressWS = () => {
      if (isCleanup) return;

      const wsUrl = `${WS_BASE_URL}/ws/research/${sessionId}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Research progress WebSocket connected');
        setConnected(true);
        onConnectionChange?.(true);
        loadSnapshot();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'research_event' || data.type === 'event_replay') {
            const evt = data.event as ProgressEvent;
            pushEvent(evt);
          } else if (data.type === 'journal_entry') {
            const entry = data.entry as JournalEntry;
            pushJournalEntry(entry);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('Research progress WebSocket disconnected');
        setConnected(false);
        onConnectionChange?.(false);

        // Only reconnect if not a clean close and component hasn't unmounted
        if (!isCleanup && event.code !== 1000) {
          progressReconnectTimeout = setTimeout(connectProgressWS, 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('Research progress WebSocket error:', error);
      };

      wsRef.current = ws;
    };

    // Connect to engine status WebSocket
    const connectEngineWS = () => {
      if (isCleanup) return;

      const wsUrl = `${WS_BASE_URL}/ws/engines/status`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Engine status WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'engine_status') {
            const status = data.status as EngineStatus;
            setEngineStatuses(prev => ({
              ...prev,
              [status.engine_name]: status,
            }));
          }
        } catch (error) {
          console.error('Error parsing engine status message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('Engine status WebSocket disconnected');

        // Only reconnect if not a clean close and component hasn't unmounted
        if (!isCleanup && event.code !== 1000) {
          engineReconnectTimeout = setTimeout(connectEngineWS, 3000);
        }
      };

      engineWsRef.current = ws;
    };

    loadSnapshot();
    connectProgressWS();
    connectEngineWS();

    return () => {
      isCleanup = true;

      // Clear any pending reconnection timeouts
      if (progressReconnectTimeout) {
        clearTimeout(progressReconnectTimeout);
      }
      if (engineReconnectTimeout) {
        clearTimeout(engineReconnectTimeout);
      }

      // Close WebSocket connections with clean close code
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000, 'Component unmounting');
      }
      if (engineWsRef.current?.readyState === WebSocket.OPEN) {
        engineWsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, [sessionId, onConnectionChange]);

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'research_started': return '🚀';
      case 'research_progress': return '⚡';
      case 'research_completed': return '✅';
      case 'research_error': return '❌';
      case 'engine_status': return '🔧';
      case 'openhands_output': return '💻';
      default: return '📋';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      case 'idle': return 'bg-gray-500';
      default: return 'bg-yellow-500';
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'destructive';
      case 'warning': return 'secondary';
      case 'success': return 'default';
      default: return 'outline';
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      {/* Engine Status Panel */}
      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            🔧 Engine Status
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Overall Progress */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Overall Progress</span>
                <span>{Math.round(overallProgress)}%</span>
              </div>
              <Progress value={overallProgress} className="w-full" />
            </div>

            <Separator />

            {/* Individual Engine Statuses */}
            <div className="space-y-3">
              {Object.values(engineStatuses).map((engine) => (
                <div key={engine.engine_name} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-medium capitalize">
                      {engine.engine_name.replace('_', ' ')}
                    </span>
                    <Badge
                      variant={engine.status === 'running' ? 'default' : 'secondary'}
                      className={`${getStatusColor(engine.status)} text-white`}
                    >
                      {engine.status}
                    </Badge>
                  </div>

                  {engine.current_task && (
                    <p className="text-sm text-muted-foreground truncate">
                      {engine.current_task}
                    </p>
                  )}

                  <Progress value={engine.progress_percentage} className="h-2" />
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Research Events Stream */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            📊 Research Progress Stream
            <Badge variant="outline">{events.length} events</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px]">
            <div className="space-y-3">
              {events.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  <div className="text-4xl mb-2">🔍</div>
                  <p>Waiting for research to begin...</p>
                </div>
              ) : (
                events.map((event, index) => (
                  <div key={index} className="flex gap-3 p-3 rounded-lg border bg-card">
                    <div className="text-xl">{getEventIcon(event.event_type)}</div>
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {event.source}
                        </Badge>
                        {event.progress_percentage !== undefined && (
                          <Badge variant="secondary" className="text-xs">
                            {Math.round(event.progress_percentage)}%
                          </Badge>
                        )}
                        <span className="text-xs text-muted-foreground">
                          {new Date(event.timestamp).toLocaleTimeString()}
                        </span>
                      </div>

                      {event.message && (
                        <p className="text-sm">{event.message}</p>
                      )}

                      {event.data.phase && (
                        <p className="text-xs text-muted-foreground">
                          Phase: {event.data.phase}
                        </p>
                      )}
                    </div>
                  </div>
                ))
              )}
              <div ref={eventsEndRef} />
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Research Journal */}
      <Card className="lg:col-span-3">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            📝 Research Journal
            <Badge variant="outline">{journalEntries.length} entries</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px]">
            <div className="space-y-2">
              {journalEntries.length === 0 ? (
                <div className="text-center text-muted-foreground py-4">
                  <p>Research journal will appear here...</p>
                </div>
              ) : (
                journalEntries.map((entry, index) => (
                  <div key={index} className="flex gap-3 p-2 rounded border-l-4 border-l-blue-500">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant="outline" className="text-xs">
                          {entry.engine}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {entry.phase}
                        </Badge>
                        <Badge variant={getLevelColor(entry.level)} className="text-xs">
                          {entry.level}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {new Date(entry.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm">{entry.message}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
};
