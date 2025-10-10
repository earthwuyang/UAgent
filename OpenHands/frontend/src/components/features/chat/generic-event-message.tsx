import React from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { code } from "../markdown/code";
import { ol, ul } from "../markdown/list";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { SuccessIndicator } from "./success-indicator";
import { ObservationResultStatus } from "./event-content-helpers/get-observation-result";
import { cn } from "#/utils/utils";

interface GenericEventMessageProps {
  title: React.ReactNode;
  details: string | React.ReactNode;
  success?: ObservationResultStatus;
  initiallyExpanded?: boolean;
  variant?: 'default' | 'sub-agent';
}

export function GenericEventMessage({
  title,
  details,
  success,
  initiallyExpanded = false,
  variant = 'default',
}: GenericEventMessageProps) {
  const [showDetails, setShowDetails] = React.useState(initiallyExpanded);

  return (
    <div 
      className={cn(
        "flex flex-col gap-2 pl-2 my-2 py-2 border-l-2 text-sm w-full",
        variant === 'sub-agent' ? 'border-purple-500' : 'border-neutral-300'
      )}
    >
      <div className="flex items-center justify-between font-bold text-neutral-300">
        <div className="flex items-center gap-2">
          {variant === 'sub-agent' && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">
              Sub-agent
            </span>
          )}
          <span>{title}</span>
          {details && (
            <button
              type="button"
              onClick={() => setShowDetails((prev) => !prev)}
              className="cursor-pointer text-left"
            >
              {showDetails ? (
                <ArrowUp className="h-4 w-4 inline fill-neutral-300" />
              ) : (
                <ArrowDown className="h-4 w-4 inline fill-neutral-300" />
              )}
            </button>
          )}
        </div>

        {success && <SuccessIndicator status={success} />}
      </div>

      {showDetails &&
        (typeof details === "string" ? (
          <Markdown
            remarkPlugins={[remarkGfm, remarkBreaks]}
            className="markdown-body"
            components={{
              code,
              ol,
              ul,
            }}
          >
            {details}
          </Markdown>
        ) : (
          details
        ))}
    </div>
  );
}
