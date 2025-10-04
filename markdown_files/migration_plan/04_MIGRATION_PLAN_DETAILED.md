# Detailed Migration Plan: UAgent → OpenHands Integration

## Overview

This document provides a detailed, phase-by-phase migration plan to integrate UAgent's research capabilities into OpenHands.

**Total Duration**: 15-20 weeks
**Approach**: Plugin/Extension Model (see Executive Summary)
**Strategy**: Gradual migration with dual-support period

---

## Phase 0: Preparation & Analysis (Weeks 1-2)

### Goals
- Finalize architecture decisions
- Set up development environment
- Establish communication with OpenHands team

### Tasks

#### Week 1: Deep Dive & Prototyping

**Day 1-2: OpenHands Analysis**
- [ ] Clone and build OpenHands from source
- [ ] Study extension/plugin system
- [ ] Identify agent creation patterns
- [ ] Review runtime architecture
- [ ] Analyze event system
- [ ] Study WebSocket implementation

**Day 3-4: Prototype Simple Integration**
- [ ] Create minimal custom agent in OpenHands
- [ ] Test custom action execution
- [ ] Verify runtime extension capability
- [ ] Test frontend component injection
- [ ] Validate WebSocket event flow

**Day 5: Documentation & Decisions**
- [ ] Document findings
- [ ] Identify any blockers
- [ ] Finalize integration approach
- [ ] Create detailed technical spec

#### Week 2: Environment Setup

**Day 1-2: Development Infrastructure**
- [ ] Fork OpenHands repository
- [ ] Set up development branch strategy
- [ ] Configure CI/CD for merged codebase
- [ ] Set up testing environments
- [ ] Create migration tracking tools

**Day 3-4: Team Alignment**
- [ ] Meet with OpenHands maintainers (if applicable)
- [ ] Present integration plan
- [ ] Get feedback on approach
- [ ] Identify collaboration opportunities
- [ ] Establish communication channels

**Day 5: Project Setup**
- [ ] Create project roadmap
- [ ] Set up issue tracking
- [ ] Define success metrics
- [ ] Create testing plan
- [ ] Document development workflow

### Deliverables
- ✅ Prototype integration working
- ✅ Technical specification finalized
- ✅ Development environment ready
- ✅ Project plan approved

---

## Phase 1: Core Backend Integration (Weeks 3-6)

### Goals
- Integrate research engines as OpenHands agents
- Extend runtime for experiment execution
- Set up data models and storage

### Week 3: Agent Foundation

**Research Agent Base Class**

```python
# openhands_extensions/research/agents/base.py

from openhands.controller.agent import Agent
from openhands.core.schema import ActionType

class ResearchAgent(Agent):
    """Base class for research-focused agents"""

    VERSION = "1.0"
    research_engine: Optional[ResearchEngine] = None

    async def step(self, state: State) -> Action:
        """Override to add research capabilities"""
        # Check if this is a research task
        if self._is_research_task(state):
            return await self._research_step(state)
        else:
            # Fall back to standard agent behavior
            return await super().step(state)

    async def _research_step(self, state: State) -> Action:
        """Research-specific step logic"""
        plan = await self.research_engine.generate_plan(state)
        return self._convert_plan_to_action(plan)
```

**Tasks**:
- [ ] Create `ResearchAgent` base class
- [ ] Implement research task detection
- [ ] Add research-specific actions
- [ ] Test agent registration with OpenHands
- [ ] Verify agent selection logic

### Week 4: Scientific Research Agent

**Port Scientific Research Engine**

```python
# openhands_extensions/research/agents/scientific.py

from .base import ResearchAgent
from ..engines.scientific_research import ScientificResearchEngine

class ScientificResearchAgent(ResearchAgent):
    """Agent for scientific experiments and hypothesis testing"""

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.research_engine = ScientificResearchEngine(llm)
        self.experiment_manager = ExperimentManager()

    async def _research_step(self, state: State) -> Action:
        # Determine research phase
        if not state.research_context:
            # Phase 1: Idea generation
            return await self._generate_ideas(state)
        elif state.research_context.ideas and not state.research_context.hypotheses:
            # Phase 2: Hypothesis formulation
            return await self._formulate_hypotheses(state)
        elif state.research_context.hypotheses:
            # Phase 3: Experiment execution
            return await self._execute_experiments(state)
        else:
            # Phase 4: Analysis
            return await self._analyze_results(state)
```

**Tasks**:
- [ ] Port `ScientificResearchEngine` class
- [ ] Integrate with OpenHands runtime
- [ ] Adapt experiment execution to use OpenHands actions
- [ ] Implement result validation
- [ ] Add tests

### Week 5: ROMA Tree Integration

**ROMA Agent for Tree-Based Research**

```python
# openhands_extensions/research/agents/roma.py

class ROMAAgent(ResearchAgent):
    """Agent for hierarchical tree-based research (ROMA)"""

    def __init__(self, llm: LLM):
        super().__init__(llm)
        self.tree_manager = ResearchTreeManager()
        self.parallel_executor = ParallelExecutor()

    async def _research_step(self, state: State) -> Action:
        # Get current node in research tree
        current_node = self.tree_manager.get_current_node(state)

        if current_node.is_leaf():
            # Execute leaf node experiment
            return await self._execute_leaf_experiment(current_node)
        else:
            # Expand tree or select child node
            return await self._expand_or_traverse(current_node)

    async def _expand_or_traverse(self, node):
        # Generate child hypotheses
        children = await self.generate_child_hypotheses(node)

        # Create parallel experiment actions
        actions = [
            ExperimentAction(child) for child in children
        ]

        # Execute in parallel using OpenHands runtime
        return ParallelAction(actions)
```

**Tasks**:
- [ ] Port ROMA tree logic
- [ ] Implement parallel execution with OpenHands
- [ ] Add tree visualization data generation
- [ ] Create tree persistence layer
- [ ] Test hierarchical research flow

### Week 6: Runtime Extensions

**Custom Action Types**

```python
# openhands_extensions/research/actions/experiment.py

from openhands.events.action import Action

class ExperimentAction(Action):
    """Action for executing research experiments"""

    action: str = "run_experiment"
    experiment_design: ExperimentDesign
    workspace_path: str
    timeout: int = 3600

    async def execute(self, controller):
        # Use OpenHands runtime to execute experiment
        runtime = controller.runtime

        # Prepare experiment environment
        await runtime.run_action(
            CmdRunAction(f"cd {self.workspace_path}")
        )

        # Execute experiment steps
        for step in self.experiment_design.steps:
            result = await runtime.run_action(
                CmdRunAction(step.command)
            )
            self.collect_result(result)

        return ExperimentObservation(
            success=self.all_steps_succeeded(),
            data=self.collected_data
        )
```

**Tasks**:
- [ ] Define custom action types (ExperimentAction, etc.)
- [ ] Create custom observation types
- [ ] Extend runtime to handle custom actions
- [ ] Implement experiment result parsing
- [ ] Add experiment workspace management

### Deliverables
- ✅ Research agents working in OpenHands
- ✅ Experiments can be executed via OpenHands runtime
- ✅ ROMA tree logic functional
- ✅ Custom actions integrated

---

## Phase 2: Data & State Management (Weeks 7-8)

### Goals
- Migrate experiment storage
- Extend OpenHands state for research data
- Implement research session management

### Week 7: State Extensions

**Extended State Model**

```python
# openhands_extensions/research/state/research_state.py

from openhands.controller.state.state import State
from dataclasses import dataclass, field

@dataclass
class ResearchContext:
    """Research-specific context"""
    research_goal: str
    ideas: List[ResearchIdea] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    tree: Optional[ResearchTree] = None
    current_phase: str = "idea_generation"

class ResearchState(State):
    """Extended state with research context"""
    research_context: Optional[ResearchContext] = None

    def add_idea(self, idea: ResearchIdea):
        if not self.research_context:
            self.research_context = ResearchContext(research_goal=idea.goal)
        self.research_context.ideas.append(idea)

    def add_experiment(self, experiment: Experiment):
        self.research_context.experiments.append(experiment)
        self.emit_event(ExperimentAddedEvent(experiment))
```

**Tasks**:
- [ ] Define research state extensions
- [ ] Implement state serialization
- [ ] Add state migration from UAgent format
- [ ] Create state validation
- [ ] Test state persistence

### Week 8: Storage & Persistence

**Experiment Storage**

```python
# openhands_extensions/research/storage/experiment_store.py

class ExperimentStore:
    """Persistent storage for experiments and results"""

    def __init__(self, storage_path: Path):
        self.db = TinyDB(storage_path / "experiments.json")
        self.workspace_root = storage_path / "workspaces"

    async def save_experiment(self, experiment: Experiment) -> str:
        exp_id = generate_id()
        exp_data = {
            "id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "design": experiment.design.dict(),
            "status": experiment.status,
            "results": experiment.results.dict() if experiment.results else None
        }
        self.db.insert(exp_data)

        # Save workspace snapshot
        workspace_path = self.workspace_root / exp_id
        await self._snapshot_workspace(experiment.workspace, workspace_path)

        return exp_id

    async def load_experiment(self, exp_id: str) -> Experiment:
        data = self.db.get(Query().id == exp_id)
        return Experiment.from_dict(data)
```

**Tasks**:
- [ ] Implement experiment storage
- [ ] Add result archival
- [ ] Create workspace snapshotting
- [ ] Implement search/query API
- [ ] Add migration from UAgent storage format

### Deliverables
- ✅ Research state integrated with OpenHands
- ✅ Experiment storage working
- ✅ Data migration tools ready

---

## Phase 3: Frontend Integration (Weeks 9-12)

### Goals
- Add research UI components to OpenHands frontend
- Implement ROMA tree visualizer
- Create experiment dashboard
- Enhance chat interface

### Week 9: Component Foundation

**Research Dashboard Component**

```typescript
// frontend/src/features/research/ResearchDashboard.tsx

import { useResearchState } from '#/hooks/useResearchState';
import { ExperimentList } from './components/ExperimentList';
import { ResearchTree } from './components/ResearchTree';

export function ResearchDashboard() {
  const { experiments, tree, phase } = useResearchState();

  return (
    <div className="research-dashboard">
      <header>
        <h1>Research Session</h1>
        <PhaseIndicator current={phase} />
      </header>

      <div className="layout-grid">
        {/* Tree visualization (if ROMA mode) */}
        {tree && (
          <div className="tree-panel">
            <ResearchTree tree={tree} />
          </div>
        )}

        {/* Experiment list */}
        <div className="experiments-panel">
          <ExperimentList experiments={experiments} />
        </div>

        {/* Real-time output */}
        <div className="output-panel">
          <LiveOutput />
        </div>
      </div>
    </div>
  );
}
```

**Tasks**:
- [ ] Create research dashboard layout
- [ ] Implement experiment list component
- [ ] Add phase indicator
- [ ] Create live output viewer
- [ ] Style with OpenHands design system

### Week 10: ROMA Tree Visualizer

**Tree Visualization**

```typescript
// frontend/src/features/research/components/ResearchTree.tsx

import { Tree } from 'react-d3-tree';

interface ResearchTreeProps {
  tree: ResearchTree;
  onNodeClick?: (node: TreeNode) => void;
}

export function ResearchTree({ tree, onNodeClick }: ResearchTreeProps) {
  const treeData = convertToD3Format(tree);

  const renderCustomNode = ({ nodeDatum }) => (
    <g>
      {/* Node visualization */}
      <circle
        r={15}
        fill={getNodeColor(nodeDatum.status)}
        onClick={() => onNodeClick?.(nodeDatum)}
      />

      {/* Node label */}
      <text y={-20}>{nodeDatum.name}</text>

      {/* Status badge */}
      <foreignObject x={20} y={-10} width={60} height={20}>
        <StatusBadge status={nodeDatum.status} />
      </foreignObject>
    </g>
  );

  return (
    <div className="tree-container">
      <Tree
        data={treeData}
        orientation="vertical"
        renderCustomNodeElement={renderCustomNode}
        pathFunc="step"
      />
    </div>
  );
}
```

**Tasks**:
- [ ] Implement tree data conversion
- [ ] Create custom node rendering
- [ ] Add node interaction (click, hover)
- [ ] Implement tree layout algorithms
- [ ] Add real-time tree updates

### Week 11: Chat Interface Enhancement

**Research Commands in Chat**

```typescript
// frontend/src/features/chat/ChatInput.tsx (enhanced)

function ChatInput() {
  const handleCommand = (command: string) => {
    // Detect research commands
    if (command.startsWith('/research')) {
      return handleResearchCommand(command);
    } else if (command.startsWith('/experiment')) {
      return handleExperimentCommand(command);
    } else if (command.startsWith('/roma')) {
      return handleRomaCommand(command);
    }

    // Standard chat handling
    return handleChatMessage(command);
  };

  const handleResearchCommand = async (cmd: string) => {
    const goal = cmd.replace('/research', '').trim();

    // Switch to research mode
    await startResearchSession({ goal });

    // Navigate to research dashboard
    navigate('/research');
  };

  // ...
}
```

**Tasks**:
- [ ] Add research command detection
- [ ] Implement `/research` command
- [ ] Add `/experiment` command
- [ ] Create `/roma` command
- [ ] Add command autocomplete

### Week 12: Integration & Polish

**Final Frontend Tasks**:
- [ ] Integrate all components with OpenHands routing
- [ ] Add navigation between chat and research views
- [ ] Implement shared state management
- [ ] Add keyboard shortcuts
- [ ] Polish UI/UX
- [ ] Add accessibility features
- [ ] Create user documentation

### Deliverables
- ✅ Research dashboard functional
- ✅ ROMA tree visualizer working
- ✅ Chat commands integrated
- ✅ UI polished and accessible

---

## Phase 4: Testing & Quality (Weeks 13-14)

### Goals
- Comprehensive testing of all features
- Performance optimization
- Bug fixes
- Documentation

### Week 13: Testing

**Unit Tests**
- [ ] Research agent tests
- [ ] ROMA tree logic tests
- [ ] State management tests
- [ ] Storage layer tests
- [ ] Action execution tests

**Integration Tests**
- [ ] End-to-end research flow
- [ ] Parallel experiment execution
- [ ] Tree-based research workflow
- [ ] Data persistence
- [ ] Frontend-backend integration

**Performance Tests**
- [ ] Large research tree handling
- [ ] Many parallel experiments
- [ ] Real-time update performance
- [ ] Storage performance
- [ ] Frontend rendering performance

### Week 14: Optimization & Bug Fixes

**Optimization**:
- [ ] Profile bottlenecks
- [ ] Optimize tree rendering
- [ ] Improve experiment execution speed
- [ ] Reduce memory usage
- [ ] Optimize WebSocket communication

**Bug Fixes**:
- [ ] Triage all issues
- [ ] Fix critical bugs
- [ ] Address user feedback
- [ ] Resolve edge cases

### Deliverables
- ✅ All tests passing
- ✅ Performance targets met
- ✅ Critical bugs fixed

---

## Phase 5: Migration & Launch (Weeks 15-16+)

### Goals
- Migrate existing UAgent users
- Launch integrated platform
- Provide ongoing support

### Week 15: Migration Tools

**Data Migration**
```python
# tools/migrate_uagent_data.py

class UAgentDataMigrator:
    """Migrate data from UAgent to integrated OpenHands"""

    async def migrate_experiments(self, uagent_db_path: Path):
        # Load UAgent experiments
        uagent_db = TinyDB(uagent_db_path)
        experiments = uagent_db.all()

        # Convert to OpenHands format
        for exp in experiments:
            openhands_exp = self.convert_experiment(exp)
            await experiment_store.save_experiment(openhands_exp)

    async def migrate_research_sessions(self, uagent_sessions_path: Path):
        # Migrate session data
        ...
```

**Tasks**:
- [ ] Create data migration scripts
- [ ] Test migration on sample data
- [ ] Add migration validation
- [ ] Create rollback mechanism
- [ ] Document migration process

### Week 16: Launch Preparation

**Documentation**:
- [ ] User guide for researchers
- [ ] Developer documentation
- [ ] API reference
- [ ] Migration guide for UAgent users
- [ ] Troubleshooting guide

**Communication**:
- [ ] Announcement blog post
- [ ] Demo video
- [ ] Tutorial series
- [ ] FAQ
- [ ] Release notes

**Deployment**:
- [ ] Deploy to staging environment
- [ ] Final integration tests
- [ ] Deploy to production
- [ ] Monitor for issues
- [ ] Provide user support

### Deliverables
- ✅ Migration tools ready
- ✅ Documentation complete
- ✅ Platform launched
- ✅ Users migrated

---

## Post-Launch (Ongoing)

### Continuous Improvement
- Monitor user feedback
- Fix bugs as reported
- Add requested features
- Keep up with OpenHands updates
- Expand research capabilities

### Future Enhancements
- More research agent types
- Enhanced visualizations
- Better parallel execution
- Advanced experiment design
- Integration with more tools

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| OpenHands API changes | Medium | High | Work with maintainers, use stable APIs |
| Performance degradation | Medium | Medium | Extensive benchmarking, optimization |
| Data migration issues | Low | High | Thorough testing, rollback plan |
| Frontend complexity | Low | Medium | Incremental development, code reviews |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|---------|------------|
| Underestimated complexity | Medium | High | Buffer time in schedule, regular reviews |
| Resource unavailability | Low | Medium | Cross-training, clear documentation |
| Dependency delays | Low | Medium | Parallel workstreams, early prototyping |

---

## Success Metrics

### Functional Metrics
- ✅ All UAgent research features available
- ✅ < 5% performance degradation vs standalone
- ✅ > 90% user satisfaction

### Technical Metrics
- ✅ > 80% code coverage
- ✅ < 10% code duplication vs standalone UAgent
- ✅ < 1s page load time

### Migration Metrics
- ✅ > 95% successful data migrations
- ✅ < 1 week avg migration time per user
- ✅ < 5% rollback rate

---

## Conclusion

This migration plan provides a structured, low-risk path to integrating UAgent's powerful research capabilities into OpenHands. The phased approach allows for:

1. ✅ **Early validation** through prototyping
2. ✅ **Incremental progress** with clear milestones
3. ✅ **Risk mitigation** through parallel tracks
4. ✅ **Quality assurance** through dedicated testing phase
5. ✅ **Smooth migration** for existing users

**Next Steps**: Review and approve, then proceed with Phase 0 prototype!

---

**Document Version**: 1.0
**Last Updated**: 2025-10-04
**Status**: Draft for Review
