/**
 * Node Event Timeline Component
 *
 * Displays a chronological list of events for a research node.
 * Features expandable event details, type-based styling, and auto-scroll.
 */

import React, { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Clock,
  AlertCircle,
  Info,
  CheckCircle,
  XCircle,
  Zap,
} from "lucide-react";
import { cn } from "#/utils/utils";
import type { ResearchNodeEvent } from "#/state/research-tree-store";

interface NodeEventTimelineProps {
  events: ResearchNodeEvent[];
  autoScroll?: boolean;
}

export function NodeEventTimeline({
  events,
  autoScroll = true,
}: NodeEventTimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  const prevEventsLengthRef = useRef(events.length);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (
      autoScroll &&
      timelineRef.current &&
      events.length > prevEventsLengthRef.current
    ) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
    prevEventsLengthRef.current = events.length;
  }, [events.length, autoScroll]);

  const toggleExpanded = (eventId: string) => {
    setExpandedEvents((prev) => {
      const next = new Set(prev);
      if (next.has(eventId)) {
        next.delete(eventId);
      } else {
        next.add(eventId);
      }
      return next;
    });
  };

  // Get icon and colors based on event type
  const getEventStyle = (eventType: string) => {
    const type = eventType.toLowerCase();

    if (type.includes("error") || type.includes("fail")) {
      return {
        icon: XCircle,
        bgColor: "bg-red-500/10 dark:bg-red-500/20",
        borderColor: "border-red-500/30",
        textColor: "text-red-700 dark:text-red-300",
        badgeBg: "bg-red-500/20",
        badgeText: "text-red-700 dark:text-red-200",
        iconColor: "text-red-500",
      };
    }

    if (type.includes("success") || type.includes("complete")) {
      return {
        icon: CheckCircle,
        bgColor: "bg-green-500/10 dark:bg-green-500/20",
        borderColor: "border-green-500/30",
        textColor: "text-green-700 dark:text-green-300",
        badgeBg: "bg-green-500/20",
        badgeText: "text-green-700 dark:text-green-200",
        iconColor: "text-green-500",
      };
    }

    if (type.includes("warning") || type.includes("warn")) {
      return {
        icon: AlertCircle,
        bgColor: "bg-yellow-500/10 dark:bg-yellow-500/20",
        borderColor: "border-yellow-500/30",
        textColor: "text-yellow-700 dark:text-yellow-300",
        badgeBg: "bg-yellow-500/20",
        badgeText: "text-yellow-700 dark:text-yellow-200",
        iconColor: "text-yellow-500",
      };
    }

    if (
      type.includes("start") ||
      type.includes("begin") ||
      type.includes("execute")
    ) {
      return {
        icon: Zap,
        bgColor: "bg-blue-500/10 dark:bg-blue-500/20",
        borderColor: "border-blue-500/30",
        textColor: "text-blue-700 dark:text-blue-300",
        badgeBg: "bg-blue-500/20",
        badgeText: "text-blue-700 dark:text-blue-200",
        iconColor: "text-blue-500",
      };
    }

    // Default: info style
    return {
      icon: Info,
      bgColor: "bg-slate-500/10 dark:bg-slate-500/20",
      borderColor: "border-slate-500/30",
      textColor: "text-slate-700 dark:text-slate-300",
      badgeBg: "bg-slate-500/20",
      badgeText: "text-slate-700 dark:text-slate-200",
      iconColor: "text-slate-500",
    };
  };

  // Format event type for display
  const formatEventType = (eventType: string) => {
    return eventType
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleString();
  };

  // Check if event has additional data to show
  const hasExpandableData = (event: ResearchNodeEvent) => {
    return Boolean(
      event.data &&
        Object.keys(event.data).length > 0 &&
        (event.data["traceback"] ||
          event.data["details"] ||
          Object.keys(event.data).length > 2),
    );
  };

  if (events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Clock size={48} className="text-slate-300 dark:text-slate-600 mb-4" />
        <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
          No Events Yet
        </h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm">
          Events will appear here as the node progresses through its execution.
        </p>
      </div>
    );
  }

  return (
    <div
      ref={timelineRef}
      className="flex-1 overflow-y-auto custom-scrollbar px-4 py-4"
    >
      <div className="max-w-4xl mx-auto space-y-3">
        {events.map((event, index) => {
          const style = getEventStyle(event.event_type);
          const Icon = style.icon;
          const isExpanded = expandedEvents.has(event.id);
          const canExpand = hasExpandableData(event);

          return (
            <div
              key={event.id}
              className={cn(
                "rounded-lg border transition-all",
                style.borderColor,
                style.bgColor,
              )}
            >
              {/* Event Header */}
              <div
                className={cn(
                  "flex items-start gap-3 p-3",
                  canExpand &&
                    "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5",
                )}
                onClick={() => canExpand && toggleExpanded(event.id)}
              >
                {/* Icon */}
                <div className="flex-shrink-0 mt-0.5">
                  <Icon size={18} className={style.iconColor} />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  {/* Type Badge + Timestamp */}
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span
                      className={cn(
                        "text-xs font-medium px-2 py-0.5 rounded",
                        style.badgeBg,
                        style.badgeText,
                      )}
                    >
                      {formatEventType(event.event_type)}
                    </span>
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      {formatTimestamp(event.timestamp)}
                    </span>
                    {event.branch_id && (
                      <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                        Branch: {event.branch_id.slice(0, 6)}
                      </span>
                    )}
                  </div>

                  {/* Message */}
                  {event.message && (
                    <p className={cn("text-sm", style.textColor)}>
                      {event.message}
                    </p>
                  )}

                  {/* Event ID */}
                  <div className="mt-1 text-xs text-slate-400 dark:text-slate-500 font-mono">
                    {event.id.slice(0, 12)}...
                  </div>
                </div>

                {/* Expand/Collapse Icon */}
                {canExpand && (
                  <div className="flex-shrink-0">
                    {isExpanded ? (
                      <ChevronDown size={18} className="text-slate-400" />
                    ) : (
                      <ChevronRight size={18} className="text-slate-400" />
                    )}
                  </div>
                )}
              </div>

              {/* Expanded Details */}
              {isExpanded && event.data && (
                <div className="border-t px-3 py-3 bg-black/5 dark:bg-white/5">
                  <h4 className="text-xs font-semibold text-slate-600 dark:text-slate-400 mb-2">
                    Event Data
                  </h4>

                  {/* Traceback (special handling) */}
                  {event.data["traceback"] && (
                    <div className="mb-3">
                      <div className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                        Traceback:
                      </div>
                      <pre className="text-xs bg-black/20 dark:bg-black/40 rounded p-2 overflow-x-auto max-h-60 overflow-y-auto">
                        {String(event.data["traceback"])}
                      </pre>
                    </div>
                  )}

                  {/* Other data fields */}
                  <div className="space-y-2">
                    {Object.entries(event.data)
                      .filter(([key]) => key !== "traceback")
                      .map(([key, value]) => {
                        const displayValue =
                          typeof value === "object"
                            ? JSON.stringify(value, null, 2)
                            : String(value);

                        return (
                          <div key={key} className="text-xs">
                            <span className="font-medium text-slate-600 dark:text-slate-400">
                              {key}:
                            </span>{" "}
                            <span className="text-slate-700 dark:text-slate-300">
                              {displayValue}
                            </span>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Event count footer */}
      <div className="text-center py-4 text-xs text-slate-500 dark:text-slate-400">
        {events.length} {events.length === 1 ? "event" : "events"} recorded
      </div>
    </div>
  );
}
