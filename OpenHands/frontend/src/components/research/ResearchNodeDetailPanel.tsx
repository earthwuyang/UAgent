import React, { useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Activity, DollarSign, Info, TrendingUp, X } from 'lucide-react';
import {
  useResearchTreeStore,
  type ResearchNode as ResearchNodeType,
} from '#/state/research-tree-store';

const detailPanelTransition = {
  type: 'spring',
  damping: 25,
  stiffness: 240,
};

export function ResearchNodeDetailPanel() {
  const nodes = useResearchTreeStore((state) => state.nodes);
  const edges = useResearchTreeStore((state) => state.edges);
  const selectedNodeId = useResearchTreeStore((state) => state.selectedNodeId);
  const selectNode = useResearchTreeStore((state) => state.selectNode);
  const getNodeEvents = useResearchTreeStore((state) => state.getNodeEvents);

  const selectedNode = useMemo<ResearchNodeType | null>(() => {
    if (!selectedNodeId) {
      return null;
    }
    return nodes.get(selectedNodeId) ?? null;
  }, [nodes, selectedNodeId]);

  const childNodes = useMemo<ResearchNodeType[]>(() => {
    if (!selectedNodeId) {
      return [];
    }
    return edges
      .filter((edge) => edge.parent_id === selectedNodeId)
      .map((edge) => nodes.get(edge.child_id))
      .filter((node): node is ResearchNodeType => Boolean(node));
  }, [edges, nodes, selectedNodeId]);

  const parentNode = useMemo<ResearchNodeType | null>(() => {
    if (!selectedNodeId) {
      return null;
    }
    const parentEdge = edges.find((edge) => edge.child_id === selectedNodeId);
    if (!parentEdge) {
      return null;
    }
    return nodes.get(parentEdge.parent_id) ?? null;
  }, [edges, nodes, selectedNodeId]);

  const nodeEvents = useMemo(
    () => (selectedNode ? getNodeEvents(selectedNode.id) : []),
    [getNodeEvents, selectedNode]
  );

  return (
    <AnimatePresence>
      {selectedNode && (
        <motion.aside
          key={selectedNode.id}
          className="research-detail-panel custom-scrollbar"
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={detailPanelTransition}
        >
          <header>
            <div>
              <h3 className="text-base font-semibold text-slate-800 dark:text-slate-100">
                {selectedNode.title}
              </h3>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                {selectedNode.type} — {selectedNode.status}
              </p>
            </div>
            <button
              type="button"
              className="header-button"
              onClick={() => selectNode(null)}
              title="Close details"
            >
              <X size={16} />
            </button>
          </header>

          <div className="detail-body custom-scrollbar">
            <section className="research-detail-section">
              <h5>Summary</h5>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                {selectedNode.content || 'No summary available yet.'}
              </p>
            </section>

            {nodeEvents.length > 0 && (
              <section className="research-detail-section">
                <h5 className="flex items-center gap-2 text-red-400">
                  Recent Errors
                  <span className="text-[10px] font-medium text-red-300">
                    {nodeEvents.length}
                  </span>
                </h5>
                <div className="flex flex-col gap-2">
                  {nodeEvents.map((event) => (
                    <div
                      key={event.id}
                      className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200"
                    >
                      <div className="font-semibold text-red-100">
                        {event.message ?? 'Execution error'}
                      </div>
                      <div className="mt-1 text-[11px] text-red-300">
                        {new Date(event.timestamp).toLocaleString()}
                      </div>
                      {event.data?.['traceback'] && (
                        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-black/20 p-2 text-[10px] text-red-100">
                          {String(event.data?.['traceback'])}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="research-detail-section">
              <h5>Metrics</h5>
              <div className="research-detail-grid">
                <div className="research-detail-metric">
                  <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                    <Activity size={16} /> Visits
                  </span>
                  <strong>{selectedNode.visits ?? 0}</strong>
                </div>
                <div className="research-detail-metric">
                  <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                    <TrendingUp size={16} /> Q Value
                  </span>
                  <strong>{(selectedNode.avg_value ?? 0).toFixed(3)}</strong>
                </div>
                <div className="research-detail-metric">
                  <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                    <Info size={16} /> Prior
                  </span>
                  <strong>{(selectedNode.prior ?? 0).toFixed(3)}</strong>
                </div>
                <div className="research-detail-metric">
                  <span className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300">
                    <DollarSign size={16} /> Cost
                  </span>
                  <strong>${(selectedNode.cost ?? 0).toFixed(3)}</strong>
                </div>
              </div>
            </section>

            <section className="research-detail-section">
              <h5>Tokens</h5>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                {(selectedNode.tokens_used ?? 0).toLocaleString()} tokens consumed.
              </p>
            </section>

            <section className="research-detail-section">
              <h5>Relationships</h5>
              <div className="research-detail-links">
                {parentNode && (
                  <button
                    type="button"
                    className="research-detail-link"
                    onClick={() => selectNode(parentNode.id)}
                  >
                    Parent: {parentNode.title}
                  </button>
                )}
                {childNodes.length > 0 ? (
                  childNodes.map((child) => (
                    <button
                      key={child.id}
                      type="button"
                      className="research-detail-link"
                      onClick={() => selectNode(child.id)}
                    >
                      Child: {child.title}
                    </button>
                  ))
                ) : (
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    No child nodes yet.
                  </p>
                )}
              </div>
            </section>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
