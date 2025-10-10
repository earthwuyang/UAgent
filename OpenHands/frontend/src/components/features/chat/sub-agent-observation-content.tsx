import React from "react";
import { useTranslation } from "react-i18next";
import {
  SubAgentSpawnedObservation,
  SubAgentProgressObservation,
  SubAgentCompletedObservation,
} from "#/types/core/observations";
import { I18nKey } from "#/i18n/declaration";

interface SubAgentSpawnedContentProps {
  event: SubAgentSpawnedObservation;
}

export function SubAgentSpawnedContent({ event }: SubAgentSpawnedContentProps) {
  const { t } = useTranslation();
  const { sub_agent_id, sub_agent_type, goal } = event.extras;

  return (
    <div className="flex flex-col gap-3 text-neutral-400">
      <div className="flex items-center gap-2">
        <span className="font-semibold">{t(I18nKey.SUB_AGENT$TYPE)}:</span>
        <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 text-xs font-mono border border-blue-500/30">
          {sub_agent_type}
        </span>
      </div>
      
      <div className="flex items-start gap-2">
        <span className="font-semibold whitespace-nowrap">ID:</span>
        <span className="font-mono text-xs text-neutral-500">{sub_agent_id}</span>
      </div>

      <div className="flex flex-col gap-1">
        <span className="font-semibold">{t(I18nKey.SUB_AGENT$GOAL)}:</span>
        <div className="text-neutral-300 pl-2 border-l-2 border-neutral-700">
          {goal.length > 300 ? (
            <details>
              <summary className="cursor-pointer hover:text-neutral-200">
                {goal.substring(0, 300)}...
              </summary>
              <div className="mt-2">{goal}</div>
            </details>
          ) : (
            goal
          )}
        </div>
      </div>
    </div>
  );
}

interface SubAgentProgressContentProps {
  event: SubAgentProgressObservation;
}

export function SubAgentProgressContent({ event }: SubAgentProgressContentProps) {
  const { t } = useTranslation();
  const { sub_agent_id, status, progress, current_task, tree_stats } = event.extras;
  const progressPercent = Math.round(progress * 100);

  return (
    <div className="flex flex-col gap-3 text-neutral-400">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="font-semibold">{t(I18nKey.SUB_AGENT$STATUS)}:</span>
          <span 
            className={`px-2 py-0.5 rounded text-xs font-medium ${
              status === 'running' 
                ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                : status === 'paused'
                ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                : 'bg-neutral-500/20 text-neutral-400 border border-neutral-500/30'
            }`}
          >
            {status}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="font-semibold">{t(I18nKey.SUB_AGENT$PROGRESS)}:</span>
          <span className="text-neutral-300 font-mono">{progressPercent}%</span>
        </div>
      </div>

      <div className="w-full bg-neutral-700 rounded-full h-2">
        <div 
          className="bg-purple-500 h-2 rounded-full transition-all duration-300"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {current_task && (
        <div className="flex flex-col gap-1">
          <span className="font-semibold">{t(I18nKey.SUB_AGENT$CURRENT_TASK)}:</span>
          <div className="text-neutral-300 pl-2 border-l-2 border-neutral-700">
            {current_task}
          </div>
        </div>
      )}

      {tree_stats && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          {tree_stats.total_nodes !== undefined && (
            <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
              <span className="text-neutral-500">Nodes:</span>
              <span className="font-mono text-neutral-300">{tree_stats.total_nodes}</span>
            </div>
          )}
          {tree_stats.total_cost !== undefined && (
            <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
              <span className="text-neutral-500">Cost:</span>
              <span className="font-mono text-neutral-300">${tree_stats.total_cost.toFixed(2)}</span>
            </div>
          )}
          {tree_stats.total_tokens !== undefined && (
            <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
              <span className="text-neutral-500">Tokens:</span>
              <span className="font-mono text-neutral-300">{tree_stats.total_tokens.toLocaleString()}</span>
            </div>
          )}
          {tree_stats.iterations !== undefined && (
            <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
              <span className="text-neutral-500">Iterations:</span>
              <span className="font-mono text-neutral-300">{tree_stats.iterations}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface SubAgentCompletedContentProps {
  event: SubAgentCompletedObservation;
}

export function SubAgentCompletedContent({ event }: SubAgentCompletedContentProps) {
  const { t } = useTranslation();
  const { sub_agent_id, status, result, tree_stats } = event.extras;

  const isSuccess = status === 'success' || status === 'completed';
  const isFailure = status === 'failed' || status === 'error';

  return (
    <div className="flex flex-col gap-3 text-neutral-400">
      <div className="flex items-center gap-2">
        <span className="font-semibold">{t(I18nKey.SUB_AGENT$STATUS)}:</span>
        <span 
          className={`px-2 py-0.5 rounded text-xs font-medium flex items-center gap-1 ${
            isSuccess
              ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
              : isFailure
              ? 'bg-red-500/20 text-red-400 border border-red-500/30'
              : 'bg-neutral-500/20 text-neutral-400 border border-neutral-500/30'
          }`}
        >
          {isSuccess && '✓'}
          {isFailure && '✗'}
          {status}
        </span>
      </div>

      <div className="flex items-start gap-2">
        <span className="font-semibold whitespace-nowrap">ID:</span>
        <span className="font-mono text-xs text-neutral-500">{sub_agent_id}</span>
      </div>

      {result && (
        <div className="flex flex-col gap-1">
          <span className="font-semibold">Result:</span>
          <div className="text-neutral-300 pl-2 border-l-2 border-neutral-700">
            {result}
          </div>
        </div>
      )}

      {tree_stats && (
        <div className="flex flex-col gap-2">
          <span className="font-semibold text-xs text-neutral-500">Tree Statistics</span>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {tree_stats.completed_nodes !== undefined && (
              <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
                <span className="text-neutral-500">Completed:</span>
                <span className="font-mono text-neutral-300">{tree_stats.completed_nodes} nodes</span>
              </div>
            )}
            {tree_stats.total_cost !== undefined && (
              <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
                <span className="text-neutral-500">Total Cost:</span>
                <span className="font-mono text-neutral-300">${tree_stats.total_cost.toFixed(3)}</span>
              </div>
            )}
            {tree_stats.total_tokens !== undefined && (
              <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
                <span className="text-neutral-500">Tokens:</span>
                <span className="font-mono text-neutral-300">{tree_stats.total_tokens.toLocaleString()}</span>
              </div>
            )}
            {tree_stats.iterations !== undefined && (
              <div className="flex justify-between px-2 py-1 bg-neutral-800/50 rounded">
                <span className="text-neutral-500">Iterations:</span>
                <span className="font-mono text-neutral-300">{tree_stats.iterations}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
