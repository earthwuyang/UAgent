import React, { useState, useCallback } from 'react';
import { Play, Pause, RotateCcw, X, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { ExperimentStatus } from '#/api/research-api';
import { startExperiment, controlExperiment, ResearchAPIError } from '#/api/research-api';
import { useResearchTreeStore } from '#/state/research-tree-store';
import { useKeyboardShortcuts } from '#/hooks/useKeyboardShortcuts';
import ConfirmDialog from './ConfirmDialog';

interface ExperimentControlsProps {
  experimentId: string;
  status: ExperimentStatus;
  onAction?: (action: string) => void;
  goal?: string;
}

const ExperimentControls: React.FC<ExperimentControlsProps> = ({
  experimentId,
  status,
  onAction,
  goal,
}) => {
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);
  const { isConnected } = useResearchTreeStore();

  const setLoading = useCallback((action: string, loading: boolean) => {
    setActionLoading(prev => ({ ...prev, [action]: loading }));
  }, []);

  const handleStart = useCallback(async () => {
    if (!experimentId) return;
    
    setLoading('start', true);
    try {
      const response = await startExperiment(experimentId, goal ?? 'Research goal');
      onAction?.('start');
      toast.success('Experiment started successfully');
      console.log('Start experiment response:', response);
    } catch (error) {
      const message = error instanceof ResearchAPIError ? error.message : 'Failed to start experiment';
      toast.error(message);
      console.error('Start experiment error:', error);
    } finally {
      setLoading('start', false);
    }
  }, [experimentId, goal, setLoading, onAction]);

  const handlePause = useCallback(async () => {
    if (!experimentId) return;
    
    setLoading('pause', true);
    try {
      await controlExperiment(experimentId, 'pause');
      onAction?.('pause');
      toast.success('Experiment paused');
    } catch (error) {
      const message = error instanceof ResearchAPIError ? error.message : 'Failed to pause experiment';
      toast.error(message);
      console.error('Pause experiment error:', error);
    } finally {
      setLoading('pause', false);
    }
  }, [experimentId, setLoading, onAction]);

  const handleResume = useCallback(async () => {
    if (!experimentId) return;
    
    setLoading('resume', true);
    try {
      await controlExperiment(experimentId, 'resume');
      onAction?.('resume');
      toast.success('Experiment resumed');
    } catch (error) {
      const message = error instanceof ResearchAPIError ? error.message : 'Failed to resume experiment';
      toast.error(message);
      console.error('Resume experiment error:', error);
    } finally {
      setLoading('resume', false);
    }
  }, [experimentId, setLoading, onAction]);

  const handleCancel = useCallback(async () => {
    if (!experimentId) return;
    
    setLoading('cancel', true);
    try {
      await controlExperiment(experimentId, 'cancel');
      onAction?.('cancel');
      toast.success('Experiment cancelled');
      setShowCancelConfirm(false);
    } catch (error) {
      const message = error instanceof ResearchAPIError ? error.message : 'Failed to cancel experiment';
      toast.error(message);
      console.error('Cancel experiment error:', error);
    } finally {
      setLoading('cancel', false);
    }
  }, [experimentId, setLoading, onAction]);

  const handleCancelClick = useCallback(() => {
    setShowCancelConfirm(true);
  }, []);

  // Keyboard shortcuts
  useKeyboardShortcuts([
    { key: 'p', ctrl: true, callback: () => status === 'running' && !actionLoading.pause && handlePause() },
    { key: 'r', ctrl: true, callback: () => status === 'paused' && !actionLoading.resume && handleResume() },
    { key: 'c', ctrl: true, shift: true, callback: () => !actionLoading.cancel && handleCancelClick() },
  ], Boolean(experimentId) && status !== 'idle');

  const isDisabled = !isConnected || Object.values(actionLoading).some(Boolean);

  return (
    <>
      <div className="flex flex-col sm:flex-row gap-2">
        {status === 'idle' && (
          <button
            onClick={handleStart}
            disabled={isDisabled || actionLoading.start}
            className="inline-flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            title="Start experiment (Ctrl+P when running)"
          >
            {actionLoading.start ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Start
          </button>
        )}

        {status === 'running' && (
          <button
            onClick={handlePause}
            disabled={isDisabled || actionLoading.pause}
            className="inline-flex items-center gap-2 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            title="Pause experiment (Ctrl+P)"
          >
            {actionLoading.pause ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Pause className="w-4 h-4" />
            )}
            Pause
          </button>
        )}

        {status === 'paused' && (
          <button
            onClick={handleResume}
            disabled={isDisabled || actionLoading.resume}
            className="inline-flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            title="Resume experiment (Ctrl+R)"
          >
            {actionLoading.resume ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RotateCcw className="w-4 h-4" />
            )}
            Resume
          </button>
        )}

        {status !== 'idle' && (
          <button
            onClick={handleCancelClick}
            disabled={isDisabled || actionLoading.cancel}
            className="inline-flex items-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            title="Cancel experiment (Ctrl+Shift+C)"
          >
            {actionLoading.cancel ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <X className="w-4 h-4" />
            )}
            Cancel
          </button>
        )}
      </div>

      <ConfirmDialog
        isOpen={showCancelConfirm}
        onClose={() => setShowCancelConfirm(false)}
        onConfirm={handleCancel}
        title="Cancel Experiment?"
        message="This will stop all running tasks. This action cannot be undone."
        variant="danger"
        confirmText="Yes, Cancel"
        isLoading={actionLoading.cancel}
      />
    </>
  );
};

export default ExperimentControls;