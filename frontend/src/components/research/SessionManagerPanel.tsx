import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';
import { Separator } from '../ui/separator';
import UAgentAPI from '../../services/api';
import type { ResearchSession, SessionListingResponse } from '../../types/api';

const REFRESH_INTERVAL_MS = 15000;

const statusClassMap: Record<string, string> = {
  pending: 'bg-yellow-500',
  running: 'bg-blue-500',
  completed: 'bg-green-500',
  error: 'bg-red-500',
};

const formatRelativeTime = (iso?: string | null): string => {
  if (!iso) return '—';
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return '—';
  const diffMs = Date.now() - parsed.getTime();
  const diffMinutes = Math.round(diffMs / 60000);
  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.round(diffHours / 24);
  return `${diffDays}d ago`;
};

const canonicalSessionId = (session: ResearchSession): string | null => (
  session.session_id ?? session.research_id ?? null
);

const SessionRow: React.FC<{ session: ResearchSession }> = ({ session }) => {
  const sessionId = canonicalSessionId(session);
  const lastActivity = session.metrics?.last_event_at || session.updated_at || session.created_at;
  const statusClass = statusClassMap[session.status?.toLowerCase?.() ?? ''] ?? 'bg-gray-500';
  const onViewSession = () => {
    if (!sessionId) return;
    const targetPath = `/session/${sessionId}`;
    window.open(targetPath, '_blank', 'noopener');
  };

  return (
    <div className="flex flex-col gap-2 rounded-lg border bg-card p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge className={`${statusClass} text-white capitalize`}>{session.status}</Badge>
          <span className="font-medium truncate" title={session.query}>{session.query}</span>
        </div>
        <Button size="sm" variant="outline" disabled={!sessionId} onClick={onViewSession}>
          View
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
        <div>
          <span className="font-semibold text-foreground">Updated:</span>
          <span className="ml-1">{formatRelativeTime(lastActivity)}</span>
        </div>
        <div>
          <span className="font-semibold text-foreground">Events:</span>
          <span className="ml-1">{session.metrics?.event_count ?? 0}</span>
        </div>
        <div>
          <span className="font-semibold text-foreground">Journal:</span>
          <span className="ml-1">{session.metrics?.journal_count ?? 0}</span>
        </div>
        <div>
          <span className="font-semibold text-foreground">LLM msgs:</span>
          <span className="ml-1">{session.metrics?.llm_message_count ?? 0}</span>
        </div>
      </div>
    </div>
  );
};

const SessionList: React.FC<{ title: string; sessions: ResearchSession[] }> = ({ title, sessions }) => (
  <div className="space-y-3">
    <div className="flex items-center justify-between">
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      <Badge variant="outline">{sessions.length}</Badge>
    </div>
    {sessions.length === 0 ? (
      <div className="rounded border border-dashed p-4 text-center text-sm text-muted-foreground">
        No sessions to display.
      </div>
    ) : (
      <ScrollArea className="max-h-72 space-y-3 pr-2">
        <div className="space-y-3">
          {sessions.map((session) => (
            <SessionRow key={canonicalSessionId(session) ?? session.query} session={session} />
          ))}
        </div>
      </ScrollArea>
    )}
  </div>
);

const SessionManagerPanel: React.FC = () => {
  const [listing, setListing] = useState<SessionListingResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await UAgentAPI.getResearchSessions();
      setListing(data);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch sessions', err);
      setError('Unable to load sessions');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
    const interval = window.setInterval(fetchSessions, REFRESH_INTERVAL_MS);
    return () => {
      window.clearInterval(interval);
    };
  }, [fetchSessions]);

  const activeSessions = useMemo<ResearchSession[]>(() => {
    if (!listing) return [];
    return (listing.active && listing.active.length > 0)
      ? listing.active
      : listing.sessions.filter((session) => session.status === 'pending' || session.status === 'running');
  }, [listing]);

  const recentSessions = useMemo<ResearchSession[]>(() => {
    if (!listing) return [];
    const base = listing.completed && listing.completed.length > 0
      ? listing.completed
      : listing.sessions.filter((session) => session.status === 'completed');
    return base.slice(0, 10);
  }, [listing]);

  const erroredSessions = useMemo<ResearchSession[]>(() => {
    if (!listing) return [];
    const base = listing.errors && listing.errors.length > 0
      ? listing.errors
      : listing.sessions.filter((session) => session.status === 'error');
    return base.slice(0, 5);
  }, [listing]);

  return (
    <Card>
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            📁 Session Manager
            <Badge variant="outline">{listing?.total ?? 0}</Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Monitor active research sessions and revisit recent results.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {error && (
            <Badge variant="destructive" className="text-xs">{error}</Badge>
          )}
          <Button size="sm" variant="outline" onClick={fetchSessions} disabled={isLoading}>
            {isLoading ? 'Refreshing…' : 'Refresh'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 lg:grid-cols-2">
          <SessionList title="Active Sessions" sessions={activeSessions} />
          <div className="space-y-4">
            <SessionList title="Recent Sessions" sessions={recentSessions} />
            <Separator />
            <SessionList title="Errored Sessions" sessions={erroredSessions} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default SessionManagerPanel;
