/**
 * Research Components
 */

/** Custom ReactFlow node renderer with expandable metrics */
export { ResearchNode } from './ResearchNode';

/** Research tree visualization with advanced controls */
export { ResearchTreeView } from './ResearchTreeView';

/** Floating container that manages the research tree experience */
export { ResearchTreePanel } from './ResearchTreePanel';

/** Detail drawer for selected research nodes */
export { ResearchNodeDetailPanel } from './ResearchNodeDetailPanel';

/** Error boundary specialised for research tree rendering */
export { ResearchErrorBoundary } from './ResearchErrorBoundary';

/** Experiment controls for starting, pausing, and cancelling research */
export { default as ExperimentControls } from './ExperimentControls';

/** Progress summary widget for research experiments */
export { default as ExperimentProgress } from './ExperimentProgress';

/** Status card for research experiments */
export { default as ExperimentStatus } from './ExperimentStatus';
