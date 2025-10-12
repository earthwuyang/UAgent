# What Happens When You Send a Research Task to OpenHands with UAgent

## Overview
When you send a research task to OpenHands with the UAgent research extension enabled, the system automatically detects complex research requests and initiates a parallel research process using the PUCT (Predictor + Upper Confidence bounds applied to Trees) algorithm. This allows the system to explore multiple research approaches simultaneously while continuing to interact with you.

## Step-by-Step Process

### 1. Message Reception
When you send a message to OpenHands:
- The message flows through the normal OpenHands processing pipeline
- The **ResearchMiddleware** intercepts the message for analysis

### 2. Task Classification
The middleware uses a **TaskClassifier** to determine if your request requires research:
- Analyzes the message content for research indicators:
  - Keywords: "research", "analyze", "investigate", "explore", "compare", etc.
  - Complexity indicators: "comprehensive", "detailed", "multiple approaches", etc.
  - Multi-step requests: tasks involving several distinct phases
- If the confidence score is above 0.5, research mode is triggered

### 3. Research Initialization
When research is triggered:
- A unique **experiment_id** is generated for tracking
- A **ResearchTree** is created with your request as the root node
- The **TreeSearchOrchestrator** is started in the background
- You receive an immediate acknowledgment that research has started

### 4. Parallel Research Execution
The orchestrator begins the PUCT loop:

#### a. Node Selection
- Uses PUCT algorithm to select the most promising node to expand
- Balances exploration (trying new approaches) with exploitation (deepening successful paths)

#### b. Node Expansion
- Generates child nodes based on the selected node type:
  - From ROOT: Generates research ideas (typically 3)
  - From IDEA: Generates hypotheses (typically 2)
  - From HYPOTHESIS: Generates experiments (typically 1)

#### c. Parallel Execution
- Executes multiple child nodes simultaneously (default: 3 parallel branches)
- Routes each task to the most appropriate adapter:
  - **DeepResearch**: Web research and information gathering
  - **RepoMaster**: Code repository analysis
  - **CodeAct**: Code execution and experiments

#### d. Results Processing
- Collects results from completed tasks
- Updates node values based on success/failure
- Continues the PUCT loop until budget is exhausted

### 5. Real-Time Updates
Throughout the process:
- Research events are streamed in real-time via WebSocket
- The research tree visualization updates dynamically
- You can monitor progress of all parallel branches
- You can send control commands (pause, resume, cancel, steer)

### 6. Completion
When research completes:
- The most successful research path is highlighted
- A comprehensive summary is generated
- All findings are organized in the research tree structure
- You can continue interacting with OpenHands normally

## Example Workflow

If you send a message like "Research and compare different neural architecture search methods":

1. **Detection**: System identifies this as a research task
2. **Initialization**: Creates research tree with "Research and compare different neural architecture search methods" as root
3. **Idea Generation**: Generates 3 research approaches:
   - Web search for recent papers
   - GitHub repository analysis
   - Implementation and benchmarking
4. **Parallel Execution**: All 3 approaches run simultaneously
5. **Results Collection**: Each branch reports findings
6. **Synthesis**: System combines findings into comprehensive comparison
7. **Presentation**: Results displayed in interactive tree format

## Key Benefits

1. **Parallel Processing**: Multiple research approaches run simultaneously
2. **Intelligent Exploration**: PUCT algorithm optimizes research strategy
3. **Non-blocking**: Research runs in background while you continue interacting
4. **Specialized Tools**: Different adapters optimized for different research tasks
5. **Real-time Monitoring**: Live updates of all research branches
6. **Budget Control**: Prevents excessive resource consumption
7. **Interactive Steering**: You can guide research in real-time

## Control Options

During research, you can:
- **Pause/Resume**: Temporarily stop or continue research
- **Cancel**: Stop research entirely
- **Steer**: Provide guidance to specific research branches
- **Reprioritize**: Adjust focus of research efforts
- **Add Nodes**: Manually add research directions

This system transforms OpenHands into a powerful research assistant that can autonomously explore complex questions while keeping you informed and in control.
