# Frontend Integration: UAgent Research UI

## Table of Contents
1. [Overview](#overview)
2. [Component Architecture](#component-architecture)
3. [Core Components](#core-components)
4. [State Management](#state-management)
5. [Styling & Theming](#styling--theming)
6. [Real-time Updates](#real-time-updates)
7. [User Workflows](#user-workflows)
8. [Accessibility](#accessibility)

---

## Overview

### Integration Strategy

The UAgent research UI will be integrated into OpenHands frontend using a **component-based approach**:

1. **Preserve OpenHands UI/UX**: Match OpenHands' design system
2. **Add Research Features**: New routes and components for research
3. **Enhance Existing Components**: Extend chat interface with research capabilities
4. **Shared State**: Integrate with OpenHands state management

### Technology Stack

```typescript
// Aligned with OpenHands frontend
- React 18+
- TypeScript 5+
- TailwindCSS (OpenHands' styling)
- Zustand or Redux (state management)
- React Router (routing)
- WebSocket (real-time updates)
- D3.js or Recharts (visualizations)
```

---

## Component Architecture

### Directory Structure

```
openhands/frontend/src/
├── components/
│   ├── research/                    # NEW: Research components
│   │   ├── dashboard/
│   │   │   ├── ResearchDashboard.tsx
│   │   │   ├── ExperimentCard.tsx
│   │   │   └── QuickActions.tsx
│   │   │
│   │   ├── experiments/
│   │   │   ├── ExperimentList.tsx
│   │   │   ├── ExperimentDetail.tsx
│   │   │   ├── ExperimentTimeline.tsx
│   │   │   ├── ProgressIndicator.tsx
│   │   │   └── ResultsViewer.tsx
│   │   │
│   │   ├── tree/
│   │   │   ├── ResearchTree.tsx
│   │   │   ├── TreeNode.tsx
│   │   │   ├── TreeBranch.tsx
│   │   │   └── TreeControls.tsx
│   │   │
│   │   ├── ideas/
│   │   │   ├── IdeaGenerator.tsx
│   │   │   ├── IdeaCard.tsx
│   │   │   └── IdeaScoring.tsx
│   │   │
│   │   ├── hypotheses/
│   │   │   ├── HypothesisPanel.tsx
│   │   │   ├── HypothesisCard.tsx
│   │   │   └── ExperimentalDesign.tsx
│   │   │
│   │   ├── code/
│   │   │   ├── CodeAnalysis.tsx
│   │   │   ├── RepoViewer.tsx
│   │   │   └── ArchitectureDiagram.tsx
│   │   │
│   │   └── shared/
│   │       ├── StatusBadge.tsx
│   │       ├── MetricCard.tsx
│   │       └── LoadingSpinner.tsx
│   │
│   └── chat/                        # MODIFIED: Enhanced with research
│       └── ChatInterface.tsx
│
├── hooks/
│   └── research/                    # NEW: Research hooks
│       ├── useExperiments.ts
│       ├── useResearchTree.ts
│       ├── useIdeas.ts
│       ├── useWebSocketStream.ts
│       └── useResearchState.ts
│
├── services/
│   └── research/                    # NEW: Research API clients
│       ├── experimentService.ts
│       ├── ideaService.ts
│       ├── codeAnalysisService.ts
│       └── websocketService.ts
│
├── store/
│   └── research/                    # NEW: Research state
│       ├── experimentsSlice.ts
│       ├── researchTreeSlice.ts
│       └── ideasSlice.ts
│
└── types/
    └── research.ts                  # NEW: Research type definitions
```

---

## Core Components

### 1. Research Dashboard

Main landing page for research features.

```typescript
// components/research/dashboard/ResearchDashboard.tsx

import React, { useState, useEffect } from 'react';
import { useExperiments } from '@/hooks/research/useExperiments';
import { ExperimentCard } from './ExperimentCard';
import { QuickActions } from './QuickActions';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { FlaskConical, GitBranch, Lightbulb, Code } from 'lucide-react';

export function ResearchDashboard() {
  const {
    activeExperiments,
    recentExperiments,
    loading,
    startExperiment
  } = useExperiments();

  const [activeTab, setActiveTab] = useState<'active' | 'recent'>('active');

  return (
    <div className="research-dashboard w-full h-full p-6 bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Research Lab
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Conduct experiments, analyze code, and explore ideas
          </p>
        </div>

        <QuickActions onActionSelect={handleQuickAction} />
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard
          icon={<FlaskConical className="w-6 h-6" />}
          label="Active Experiments"
          value={activeExperiments.length}
          color="blue"
        />
        <StatCard
          icon={<GitBranch className="w-6 h-6" />}
          label="Research Branches"
          value={researchBranches}
          color="green"
        />
        <StatCard
          icon={<Lightbulb className="w-6 h-6" />}
          label="Ideas Generated"
          value={ideasCount}
          color="yellow"
        />
        <StatCard
          icon={<Code className="w-6 h-6" />}
          label="Repos Analyzed"
          value={reposAnalyzed}
          color="purple"
        />
      </div>

      {/* Experiments Section */}
      <Card>
        <CardHeader>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="active">
                Active ({activeExperiments.length})
              </TabsTrigger>
              <TabsTrigger value="recent">
                Recent ({recentExperiments.length})
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>

        <CardContent>
          {loading ? (
            <LoadingSpinner />
          ) : (
            <div className="space-y-4">
              {activeTab === 'active' ? (
                activeExperiments.length > 0 ? (
                  activeExperiments.map(exp => (
                    <ExperimentCard
                      key={exp.id}
                      experiment={exp}
                      onViewDetails={() => navigate(`/research/experiments/${exp.id}`)}
                    />
                  ))
                ) : (
                  <EmptyState
                    icon={<FlaskConical />}
                    message="No active experiments"
                    action={
                      <Button onClick={() => setShowNewExperimentDialog(true)}>
                        Start Experiment
                      </Button>
                    }
                  />
                )
              ) : (
                recentExperiments.map(exp => (
                  <ExperimentCard
                    key={exp.id}
                    experiment={exp}
                    onViewDetails={() => navigate(`/research/experiments/${exp.id}`)}
                  />
                ))
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Supporting components
function StatCard({ icon, label, value, color }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-300',
    green: 'bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-300',
    yellow: 'bg-yellow-100 text-yellow-600 dark:bg-yellow-900 dark:text-yellow-300',
    purple: 'bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-300',
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center gap-4">
          <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
            {icon}
          </div>
          <div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {value}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {label}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

### 2. Experiment Detail View

Detailed view of a running or completed experiment.

```typescript
// components/research/experiments/ExperimentDetail.tsx

import React, { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useExperiment } from '@/hooks/research/useExperiments';
import { useWebSocketStream } from '@/hooks/research/useWebSocketStream';
import { ExperimentTimeline } from './ExperimentTimeline';
import { ProgressIndicator } from './ProgressIndicator';
import { ResultsViewer } from './ResultsViewer';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';

export function ExperimentDetail() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const { experiment, loading, error } = useExperiment(experimentId);
  const { events, connectionStatus } = useWebSocketStream(experimentId);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error || !experiment) {
    return <ErrorState error={error || 'Experiment not found'} />;
  }

  return (
    <div className="experiment-detail w-full h-full p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {experiment.goal}
            </h1>
            <div className="flex gap-3 mt-2">
              <StatusBadge status={experiment.status} />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Started {formatDistanceToNow(experiment.started_at)}
              </span>
              {experiment.status === 'running' && (
                <span className="flex items-center gap-1 text-sm">
                  <div className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                  }`} />
                  {connectionStatus === 'connected' ? 'Live' : 'Disconnected'}
                </span>
              )}
            </div>
          </div>

          <div className="flex gap-2">
            {experiment.status === 'running' && (
              <>
                <Button variant="outline" onClick={handlePause}>
                  Pause
                </Button>
                <Button variant="destructive" onClick={handleCancel}>
                  Cancel
                </Button>
              </>
            )}
            {experiment.status === 'completed' && (
              <Button onClick={handleDownloadResults}>
                Download Results
              </Button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {experiment.status === 'running' && (
          <ProgressIndicator
            progress={experiment.progress}
            className="mt-4"
          />
        )}
      </div>

      {/* Content Tabs */}
      <Tabs defaultValue="timeline" className="w-full">
        <TabsList>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="results">
            Results
            {experiment.status === 'completed' && (
              <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                Ready
              </span>
            )}
          </TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
          <TabsTrigger value="artifacts">
            Artifacts
            {experiment.artifacts?.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">
                {experiment.artifacts.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="timeline" className="mt-6">
          <ExperimentTimeline
            events={events}
            currentStep={experiment.current_step}
          />
        </TabsContent>

        <TabsContent value="results" className="mt-6">
          {experiment.status === 'completed' ? (
            <ResultsViewer results={experiment.results} />
          ) : (
            <div className="text-center py-12 text-gray-500">
              Results will be available when the experiment completes
            </div>
          )}
        </TabsContent>

        <TabsContent value="logs" className="mt-6">
          <LogViewer
            logs={experiment.logs}
            liveEvents={events}
            streaming={experiment.status === 'running'}
          />
        </TabsContent>

        <TabsContent value="artifacts" className="mt-6">
          <ArtifactsList artifacts={experiment.artifacts} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

### 3. Research Tree Visualizer (ROMA)

Interactive tree visualization for ROMA research.

```typescript
// components/research/tree/ResearchTree.tsx

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResearchTree } from '@/hooks/research/useResearchTree';
import { TreeControls } from './TreeControls';
import { TreeNode } from '@/types/research';

export function ResearchTree({ sessionId }: { sessionId: string }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const { tree, loading, updateNode } = useResearchTree(sessionId);
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (!tree || !svgRef.current) return;

    renderTree(tree, svgRef.current, {
      onNodeClick: setSelectedNode,
      zoom
    });
  }, [tree, zoom]);

  return (
    <div className="research-tree relative w-full h-full bg-white dark:bg-gray-900 rounded-lg">
      {/* Controls */}
      <TreeControls
        zoom={zoom}
        onZoomIn={() => setZoom(z => Math.min(z + 0.1, 2))}
        onZoomOut={() => setZoom(z => Math.max(z - 0.1, 0.5))}
        onReset={() => setZoom(1)}
        onExpandAll={() => expandAll(tree)}
        onCollapseAll={() => collapseAll(tree)}
      />

      {/* Tree SVG */}
      <svg
        ref={svgRef}
        className="w-full h-full"
        style={{ minHeight: '600px' }}
      />

      {/* Node Detail Panel */}
      {selectedNode && (
        <NodeDetailPanel
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
          onUpdate={updateNode}
        />
      )}
    </div>
  );
}

// D3 rendering logic
function renderTree(
  tree: ResearchTree,
  svg: SVGSVGElement,
  options: RenderOptions
) {
  const width = svg.clientWidth;
  const height = svg.clientHeight;

  // Clear previous render
  d3.select(svg).selectAll('*').remove();

  // Create tree layout
  const treeLayout = d3.tree<TreeNode>()
    .size([width - 100, height - 100]);

  // Create hierarchy
  const root = d3.hierarchy(tree.root);
  const treeData = treeLayout(root);

  // Create SVG group
  const g = d3.select(svg)
    .append('g')
    .attr('transform', `translate(50, 50) scale(${options.zoom})`);

  // Draw links
  g.selectAll('.link')
    .data(treeData.links())
    .enter()
    .append('path')
    .attr('class', 'link')
    .attr('fill', 'none')
    .attr('stroke', '#94a3b8')
    .attr('stroke-width', 2)
    .attr('d', d3.linkVertical()
      .x((d: any) => d.x)
      .y((d: any) => d.y)
    );

  // Draw nodes
  const nodes = g.selectAll('.node')
    .data(treeData.descendants())
    .enter()
    .append('g')
    .attr('class', 'node')
    .attr('transform', d => `translate(${d.x}, ${d.y})`)
    .on('click', (event, d) => options.onNodeClick(d.data));

  // Node circles
  nodes.append('circle')
    .attr('r', 20)
    .attr('fill', d => getNodeColor(d.data.status))
    .attr('stroke', '#fff')
    .attr('stroke-width', 3);

  // Node icons
  nodes.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', 5)
    .text(d => getNodeIcon(d.data.type));

  // Node labels
  nodes.append('text')
    .attr('dy', 35)
    .attr('text-anchor', 'middle')
    .attr('font-size', 12)
    .text(d => truncate(d.data.content, 30));

  // Status indicators
  nodes.filter(d => d.data.status === 'running')
    .append('circle')
    .attr('r', 25)
    .attr('fill', 'none')
    .attr('stroke', '#3b82f6')
    .attr('stroke-width', 2)
    .attr('class', 'animate-pulse');
}

function getNodeColor(status: string): string {
  const colors = {
    pending: '#94a3b8',
    running: '#3b82f6',
    completed: '#10b981',
    failed: '#ef4444'
  };
  return colors[status] || colors.pending;
}

function getNodeIcon(type: string): string {
  const icons = {
    hypothesis: '💡',
    experiment: '🔬',
    analysis: '📊'
  };
  return icons[type] || '❓';
}
```

### 4. Idea Generator

AI-powered idea generation interface.

```typescript
// components/research/ideas/IdeaGenerator.tsx

import React, { useState } from 'react';
import { useIdeas } from '@/hooks/research/useIdeas';
import { IdeaCard } from './IdeaCard';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Lightbulb, Sparkles } from 'lucide-react';

export function IdeaGenerator() {
  const { ideas, generating, generateIdeas } = useIdeas();
  const [topic, setTopic] = useState('');
  const [numIdeas, setNumIdeas] = useState(5);
  const [creativity, setCreativity] = useState(0.7);

  const handleGenerate = async () => {
    await generateIdeas({
      topic,
      num_ideas: numIdeas,
      creativity,
    });
  };

  return (
    <div className="idea-generator p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
            <Lightbulb className="w-8 h-8 text-yellow-500" />
            Idea Generator
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Generate novel research ideas using AI
          </p>
        </div>

        {/* Input Form */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <div className="space-y-6">
              {/* Topic Input */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Research Topic
                </label>
                <Textarea
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                  placeholder="e.g., Machine learning for compiler optimization"
                  rows={3}
                  className="w-full"
                />
              </div>

              {/* Number of Ideas */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Number of Ideas: {numIdeas}
                </label>
                <Slider
                  value={[numIdeas]}
                  onValueChange={([value]) => setNumIdeas(value)}
                  min={1}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </div>

              {/* Creativity */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Creativity: {creativity.toFixed(1)}
                </label>
                <Slider
                  value={[creativity]}
                  onValueChange={([value]) => setCreativity(value)}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Higher values produce more creative but potentially less feasible ideas
                </p>
              </div>

              {/* Generate Button */}
              <Button
                onClick={handleGenerate}
                disabled={!topic || generating}
                className="w-full"
                size="lg"
              >
                {generating ? (
                  <>
                    <Sparkles className="w-5 h-5 mr-2 animate-spin" />
                    Generating Ideas...
                  </>
                ) : (
                  <>
                    <Lightbulb className="w-5 h-5 mr-2" />
                    Generate Ideas
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Generated Ideas */}
        {ideas.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              Generated Ideas ({ideas.length})
            </h2>

            {ideas.map(idea => (
              <IdeaCard
                key={idea.id}
                idea={idea}
                onSelect={() => handleSelectIdea(idea)}
                onGenerateHypotheses={() => handleGenerateHypotheses(idea)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## State Management

### Research State Store

```typescript
// store/research/experimentsSlice.ts

import { create } from 'zustand';
import { Experiment, ExperimentStatus } from '@/types/research';
import { experimentService } from '@/services/research/experimentService';

interface ExperimentsState {
  experiments: Record<string, Experiment>;
  loading: boolean;
  error: string | null;

  // Actions
  fetchExperiments: (sessionId: string) => Promise<void>;
  fetchExperiment: (experimentId: string) => Promise<void>;
  startExperiment: (config: StartExperimentConfig) => Promise<string>;
  cancelExperiment: (experimentId: string) => Promise<void>;
  updateExperiment: (experimentId: string, updates: Partial<Experiment>) => void;
}

export const useExperimentsStore = create<ExperimentsState>((set, get) => ({
  experiments: {},
  loading: false,
  error: null,

  fetchExperiments: async (sessionId: string) => {
    set({ loading: true, error: null });
    try {
      const experiments = await experimentService.list({ session_id: sessionId });
      const experimentsMap = experiments.reduce((acc, exp) => {
        acc[exp.id] = exp;
        return acc;
      }, {} as Record<string, Experiment>);

      set({ experiments: experimentsMap, loading: false });
    } catch (error) {
      set({ error: error.message, loading: false });
    }
  },

  fetchExperiment: async (experimentId: string) => {
    try {
      const experiment = await experimentService.get(experimentId);
      set(state => ({
        experiments: {
          ...state.experiments,
          [experimentId]: experiment
        }
      }));
    } catch (error) {
      set({ error: error.message });
    }
  },

  startExperiment: async (config: StartExperimentConfig) => {
    set({ loading: true, error: null });
    try {
      const experiment = await experimentService.start(config);
      set(state => ({
        experiments: {
          ...state.experiments,
          [experiment.id]: experiment
        },
        loading: false
      }));
      return experiment.id;
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },

  cancelExperiment: async (experimentId: string) => {
    try {
      await experimentService.cancel(experimentId);
      set(state => ({
        experiments: {
          ...state.experiments,
          [experimentId]: {
            ...state.experiments[experimentId],
            status: 'cancelled' as ExperimentStatus
          }
        }
      }));
    } catch (error) {
      set({ error: error.message });
    }
  },

  updateExperiment: (experimentId: string, updates: Partial<Experiment>) => {
    set(state => ({
      experiments: {
        ...state.experiments,
        [experimentId]: {
          ...state.experiments[experimentId],
          ...updates
        }
      }
    }));
  }
}));
```

### Custom Hooks

```typescript
// hooks/research/useExperiments.ts

import { useEffect, useMemo } from 'react';
import { useExperimentsStore } from '@/store/research/experimentsSlice';
import { useOpenHandsSession } from '@/hooks/useOpenHandsSession';

export function useExperiments() {
  const { sessionId } = useOpenHandsSession();
  const {
    experiments,
    loading,
    error,
    fetchExperiments,
    startExperiment,
    cancelExperiment
  } = useExperimentsStore();

  // Fetch experiments on mount
  useEffect(() => {
    if (sessionId) {
      fetchExperiments(sessionId);
    }
  }, [sessionId, fetchExperiments]);

  // Derived data
  const experimentsList = useMemo(
    () => Object.values(experiments),
    [experiments]
  );

  const activeExperiments = useMemo(
    () => experimentsList.filter(exp =>
      exp.status === 'running' || exp.status === 'pending'
    ),
    [experimentsList]
  );

  const completedExperiments = useMemo(
    () => experimentsList.filter(exp => exp.status === 'completed'),
    [experimentsList]
  );

  const failedExperiments = useMemo(
    () => experimentsList.filter(exp => exp.status === 'failed'),
    [experimentsList]
  );

  return {
    experiments: experimentsList,
    activeExperiments,
    completedExperiments,
    failedExperiments,
    loading,
    error,
    startExperiment,
    cancelExperiment
  };
}

// Hook for single experiment
export function useExperiment(experimentId: string) {
  const { experiments, fetchExperiment } = useExperimentsStore();

  useEffect(() => {
    fetchExperiment(experimentId);
  }, [experimentId, fetchExperiment]);

  return {
    experiment: experiments[experimentId],
    loading: !experiments[experimentId],
    error: null
  };
}
```

---

## Styling & Theming

### TailwindCSS Theme Extension

```javascript
// tailwind.config.js (extensions to OpenHands config)

module.exports = {
  theme: {
    extend: {
      colors: {
        // Research-specific colors
        research: {
          primary: '#3b82f6',
          secondary: '#8b5cf6',
          success: '#10b981',
          warning: '#f59e0b',
          danger: '#ef4444',
        },
        // Experiment status colors
        experiment: {
          pending: '#94a3b8',
          running: '#3b82f6',
          completed: '#10b981',
          failed: '#ef4444',
          timeout: '#f59e0b',
          cancelled: '#6b7280',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    }
  }
};
```

### Component Styling Patterns

```typescript
// Consistent styling patterns for research components

const buttonStyles = {
  primary: 'bg-research-primary hover:bg-blue-600 text-white',
  secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-900',
  danger: 'bg-research-danger hover:bg-red-600 text-white',
};

const cardStyles = {
  base: 'bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700',
  hover: 'hover:shadow-md transition-shadow',
};

const statusBadgeStyles = {
  pending: 'bg-experiment-pending text-white',
  running: 'bg-experiment-running text-white animate-pulse',
  completed: 'bg-experiment-completed text-white',
  failed: 'bg-experiment-failed text-white',
};
```

---

## Real-time Updates

### WebSocket Integration

```typescript
// hooks/research/useWebSocketStream.ts

import { useEffect, useState } from 'react';
import { ResearchEvent } from '@/types/research';
import { websocketService } from '@/services/research/websocketService';
import { useExperimentsStore } from '@/store/research/experimentsSlice';

export function useWebSocketStream(experimentId: string) {
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const { updateExperiment } = useExperimentsStore();

  useEffect(() => {
    if (!experimentId) return;

    setConnectionStatus('connecting');

    const ws = websocketService.connect(experimentId, {
      onOpen: () => {
        setConnectionStatus('connected');
      },
      onClose: () => {
        setConnectionStatus('disconnected');
      },
      onEvent: (event: ResearchEvent) => {
        // Add event to list
        setEvents(prev => [...prev, event]);

        // Update experiment state based on event
        if (event.type === 'experiment_progress') {
          updateExperiment(experimentId, {
            progress: event.progress
          });
        } else if (event.type === 'experiment_completed') {
          updateExperiment(experimentId, {
            status: 'completed',
            results: event.results,
            completed_at: event.timestamp
          });
        } else if (event.type === 'experiment_failed') {
          updateExperiment(experimentId, {
            status: 'failed',
            error: event.error
          });
        }
      },
      onError: (error) => {
        console.error('WebSocket error:', error);
      }
    });

    return () => {
      ws.close();
    };
  }, [experimentId, updateExperiment]);

  return {
    events,
    connectionStatus,
    isConnected: connectionStatus === 'connected'
  };
}
```

---

## User Workflows

### Workflow 1: Scientific Experiment

```
1. User clicks "New Experiment" button
   ↓
2. Opens "Start Experiment" dialog
   - Select type: Scientific Research
   - Enter goal: "Compare sorting algorithms"
   - Configure options
   ↓
3. Clicks "Start" → API call → Experiment created
   ↓
4. Navigate to Experiment Detail page
   - Shows real-time progress
   - Displays timeline of steps
   - Streams LLM interactions
   ↓
5. Experiment completes
   - Results viewer displays findings
   - Download button appears
   - Option to start related experiment
```

### Workflow 2: Code Analysis

```
1. User selects "Analyze Repository"
   ↓
2. Opens Code Analysis form
   - Enter repo URL or select local path
   - Enter analysis query
   - Select depth (quick/deep)
   ↓
3. Analysis runs
   - Shows progress indicator
   - Displays files being analyzed
   ↓
4. Results displayed
   - Architecture diagram
   - Relevant files list
   - Code map visualization
   - Insights and recommendations
   ↓
5. User can:
   - Drill down into specific files
   - Ask follow-up questions
   - Export analysis report
```

### Workflow 3: ROMA Research

```
1. User starts ROMA session
   ↓
2. Enter research topic
   - System generates initial hypotheses
   - Creates root node in tree
   ↓
3. For each hypothesis:
   - Generate sub-hypotheses (branches)
   - Run experiments in parallel
   - Update tree in real-time
   ↓
4. User interacts with tree:
   - Click nodes to see details
   - Expand/collapse branches
   - Prune low-confidence paths
   ↓
5. Research converges:
   - Synthesize findings from all branches
   - Generate comprehensive report
   - Visualize research journey
```

---

## Accessibility

### WCAG 2.1 AA Compliance

```typescript
// Accessibility features for all research components

// 1. Keyboard Navigation
const handleKeyDown = (e: React.KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    handleNodeClick();
  }
  if (e.key === 'Escape') {
    closePanel();
  }
};

// 2. ARIA Labels
<button
  aria-label="Start new scientific experiment"
  aria-describedby="experiment-help-text"
>
  Start Experiment
</button>

// 3. Focus Management
const dialogRef = useRef<HTMLDivElement>(null);

useEffect(() => {
  if (isOpen && dialogRef.current) {
    dialogRef.current.focus();
  }
}, [isOpen]);

// 4. Screen Reader Support
<div role="status" aria-live="polite" aria-atomic="true">
  {experiment.status === 'running' && (
    <span className="sr-only">
      Experiment in progress: {experiment.progress.percentage}% complete
    </span>
  )}
</div>

// 5. Color Contrast
// All status colors meet WCAG AA contrast requirements:
const statusColors = {
  pending: '#64748b',   // 4.8:1 on white
  running: '#2563eb',   // 4.5:1 on white
  completed: '#059669', // 4.5:1 on white
  failed: '#dc2626',    // 5.1:1 on white
};

// 6. Focus Indicators
.focus-visible:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}
```

---

## Performance Optimization

### Code Splitting

```typescript
// Lazy load research components
const ResearchDashboard = lazy(() => import('./components/research/dashboard/ResearchDashboard'));
const ExperimentDetail = lazy(() => import('./components/research/experiments/ExperimentDetail'));
const ResearchTree = lazy(() => import('./components/research/tree/ResearchTree'));

// Routes with Suspense
<Routes>
  <Route path="/research" element={
    <Suspense fallback={<LoadingSpinner />}>
      <ResearchDashboard />
    </Suspense>
  } />
</Routes>
```

### Memoization

```typescript
// Memoize expensive computations
const sortedExperiments = useMemo(() => {
  return experiments.sort((a, b) =>
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
}, [experiments]);

// Memoize components
const ExperimentCard = memo(({ experiment, onClick }: ExperimentCardProps) => {
  // Component implementation
}, (prevProps, nextProps) => {
  return prevProps.experiment.id === nextProps.experiment.id &&
         prevProps.experiment.status === nextProps.experiment.status;
});
```

### Virtual Scrolling

```typescript
// For long lists of experiments
import { useVirtualizer } from '@tanstack/react-virtual';

function ExperimentList({ experiments }: { experiments: Experiment[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: experiments.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100,
    overscan: 5
  });

  return (
    <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px` }}>
        {virtualizer.getVirtualItems().map(virtualRow => (
          <div
            key={virtualRow.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              transform: `translateY(${virtualRow.start}px)`
            }}
          >
            <ExperimentCard experiment={experiments[virtualRow.index]} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

**Next**: See `08_RISK_MITIGATION.md` for comprehensive risk analysis and mitigation strategies.
