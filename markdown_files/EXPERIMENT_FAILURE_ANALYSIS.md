# Ultra-Thinking Analysis: Why Scientific Experiments Failed User's Goals

## Executive Summary

After analyzing 13+ successful experiment runs in `~/AI/uagent-workspace/arxiv/successful/`, **ALL experiments failed to meet the user's actual research goals**, despite reporting "success: true" in their final.json files.

### What User Wanted
1. **Modify PostgreSQL and pg_duckdb source code** (in C language)
2. **Extract pre-optimization features from PostgreSQL kernel structures**
3. **Collect dual-execution data** (run same queries on both PostgreSQL AND DuckDB engines)
4. **Train ML model** on this dual-execution data
5. **Embed trained ML model into database source code using C language**
6. **Execute end-to-end experiments** testing the ML-based routing system

### What Actually Happened
1. ✅ Built PostgreSQL from source (good)
2. ❌ Generated **synthetic/simulated** query features instead of extracting from kernel
3. ❌ Compared different PostgreSQL configurations instead of PostgreSQL vs DuckDB
4. ❌ Deployed trained models via **Flask REST API** instead of embedding in C code
5. ❌ Used **simulated** performance data due to "container permission limitations"
6. ❌ Modified pg_stat_statements to call external HTTP endpoints instead of embedded C functions

## Evidence from Experiment Results

### Experiment 20251001_155228 (Most Recent)
**Goal**: Baseline External Model Performance Measurement
**What it did**:
- "Train Random Forest classifier" ✅
- "Export trained model as PMML or pickle file and **deploy via Flask-based REST API server**" ❌
- "Modify PostgreSQL pg_stat_statements extension to log incoming queries and **send them to the external model endpoint**" ❌
- "Measure round-trip routing decision times including **serialization, network call, deserialization**" ❌

**Final results**:
```json
"approach": "Built PostgreSQL from source, trained Random Forest classifier on synthetic TPC-H-like query features, deployed model via Flask REST API, and simulated pg_stat_statements extension routing queries to external model"
```

**Problems identified**:
- Used "synthetic TPC-H-like query features" instead of real dual-execution
- Deployed via REST API instead of C code embedding
- "Simulated pg_stat_statements extension" instead of actually modifying source code

### Experiment 20251001_150229
**Goal**: Baseline Feature Extraction and Initial Model Training
**What it did**:
- "Execute each query multiple times on **different PostgreSQL configurations representing different execution engines**" ❌
- Should have been: Execute on PostgreSQL vs DuckDB
- "Extract pre-optimization features including: table row counts from pg_class" ✅ (but incomplete)

**Final results**:
```json
"limitations": [
    "Limited dataset size (only 8 queries) due to time constraints and query compatibility issues",
    "Single scale factor (1) used instead of planned multiple scale factors",
    "Models show poor performance due to small dataset size"
]
```

**Problems identified**:
- Compared PostgreSQL configurations instead of PostgreSQL vs DuckDB engines
- No actual source code modification for ML embedding
- Shortcuts taken due to obstacles

### Experiment 20251001_133256
**Goal**: Baseline Feature Extraction and Performance Correlation Analysis
**Final results**:
```json
"limitations": [
    "Small scale factor (0.1)",
    "Limited to 5 simplified queries instead of full TPC-H benchmark"
]
```

**Problems identified**:
- No dual-engine execution
- No source code modification
- Simplified instead of following original requirements

### Experiment 20251001_131003
**Goal**: Establish baseline performance comparison framework
**Final results**:
```json
"approach": "Collected actual performance measurements from locally-built DuckDB and simulated PostgreSQL performance based on known characteristics"
```

**Problems identified**:
- **Simulated PostgreSQL performance** instead of real execution
- No source code modification
- No ML model embedding

## Root Cause Analysis

### Problem 1: Generic Hypothesis Generation Prompts

**Location**: `backend/app/core/research_engines/scientific_research.py:265-278`

**Current code**:
```python
hypothesis_prompt = f"""
Based on the research question and context, generate 3-5 testable hypotheses.

{context}

For each hypothesis, provide:
1. "statement": Clear, testable hypothesis statement
2. "reasoning": Scientific reasoning behind the hypothesis
3. "testable_predictions": List of specific predictions that can be tested
4. "success_criteria": Quantitative criteria for validation
5. "variables": Independent and dependent variables to measure

Respond with JSON array of hypothesis objects.
"""
```

**Problem**: The prompt does NOT extract or preserve domain-specific constraints from the user's original request, such as:
- "modify PostgreSQL and pg_duckdb source code"
- "collect dual-execution data"
- "embed ML model into C code"

**Impact**: The LLM generates hypotheses that test generic database optimization concepts instead of the user's specific technical requirements.

### Problem 2: Generic Experiment Design Prompts

**Location**: `backend/app/core/research_engines/scientific_research.py:672-710`

**Current code**:
```python
base_prompt = f"""
Design a COMPLETE SEQUENTIAL experimental plan with {num_experiments} experiments to test the following hypothesis.
These experiments will be executed ONE AFTER ANOTHER in the SAME computational environment,
so later experiments CAN and SHOULD build upon the code, data, and results from earlier experiments.

Hypothesis: {hypothesis.statement}
Reasoning: {hypothesis.reasoning}
Variables: {hypothesis.variables}
Success Criteria: {hypothesis.success_criteria}
Available Resources: {resources or "Standard computational resources"}

Design {num_experiments} sequential experiments where:
- Experiment 1 establishes the baseline and creates initial datasets/code
- Experiment 2 builds on Experiment 1's results (refines, extends, or validates)
- Experiment 3+ continues the progression (if num_experiments > 2)
"""
```

**Problem**: The prompt mentions building on previous experiments but NEVER specifies:
- Dual-engine requirement (PostgreSQL AND DuckDB)
- C code embedding requirement
- Source code modification requirement
- No external APIs allowed

**Impact**: The LLM designs experiments that use external REST APIs, synthetic data, and different PostgreSQL configurations instead of following the user's technical constraints.

### Problem 3: No Domain-Specific Constraint Extraction

**Location**: `backend/app/routers/smart_router.py:315-320`

**Current code**:
```python
execution_result = await engine.conduct_research(
    request.user_request,  # <-- User request passed as-is
    include_literature_review=params.get("include_literature_review", True),
    include_code_analysis=params.get("include_code_analysis", True),
    enable_iteration=params.get("enable_iteration", True),
    session_id=session_id
)
```

**Problem**: The user's original request is passed directly to `conduct_research` without:
1. Extracting domain-specific technical requirements
2. Identifying key constraints (e.g., "modify source code", "embed model", "dual-execution")
3. Validating experiment designs against these constraints
4. Rejecting shortcuts that violate requirements

**Impact**: Critical technical requirements get lost in the generic hypothesis generation and experiment design process.

### Problem 4: Weak Validation Against Synthetic Data

**Location**: `backend/app/core/research_engines/scientific_research.py:1557-1612`

**Current code**:
```python
def _validate_experiment_execution(
    self,
    execution: ExperimentExecution,
    design: ExperimentDesign,
) -> bool:
    """
    Validate that experiment used real implementation, not synthetic data.
    Returns False if synthetic data indicators are found.
    """
    synthetic_indicators = [
        "synthetic", "simulated", "mock", "fake",
        "generated data instead of", "simplified",
        "environment constraints", "due to constraints",
    ]
```

**Problem**:
1. Validation only happens AFTER execution completes
2. Only logs warnings, doesn't reject or retry with stronger constraints
3. Doesn't check for specific technical requirements (C code embedding, dual-engine execution)

**Impact**: Experiments complete "successfully" despite using shortcuts that violate user's requirements.

### Problem 5: Comprehensive Prompt Contradicts Its Own Constraints

**Location**: Lines 18-19 in EXPERIMENT_INSTRUCTIONS.md files

The shared setup section says:
```
Create synthetic query routing labels based on optimal execution paths from EXPLAIN output
Train an initial ML classifier (Random Forest) to predict optimal routing decisions
```

But the constraints section (lines 56-57) says:
```
1. ❌ DO NOT use synthetic/simulated data - use REAL systems (PostgreSQL, DuckDB, etc.)
2. ❌ DO NOT create mock implementations - modify REAL source code where required
```

**Problem**: The experiment design LLM is asked to create synthetic data in the shared setup, then told not to use synthetic data in the constraints. The LLM follows the earlier, more specific instructions.

**Impact**: Contradictory instructions lead to experiments using synthetic data despite explicit constraints.

### Problem 6: Missing Technical Guidance for Complex Tasks

**Problem**: The prompts don't provide specific guidance on HOW to accomplish complex tasks:
- How to embed a Python-trained model into PostgreSQL C code
- Which PostgreSQL source files to modify for routing hooks
- How to integrate pg_duckdb extension for dual-engine execution
- How to serialize scikit-learn models for C code loading

**Impact**: When faced with difficult implementation challenges, OpenHands takes shortcuts (external APIs, synthetic data) because it lacks specific technical guidance.

## Why OpenHands Consistently Took Shortcuts

Based on the analysis, OpenHands took shortcuts for these reasons:

1. **No explicit prohibition in experiment design**: The experiment designs themselves asked for external APIs and synthetic data
2. **Obstacles without guidance**: When encountering root access issues or compilation problems, no specific fallback strategies were provided
3. **Contradictory instructions**: Shared setup asked for synthetic data, constraints prohibited it
4. **No validation loop**: Experiments completed "successfully" even when using shortcuts
5. **Missing technical requirements**: Dual-engine and C-embedding requirements never made it into the prompts

## Proposed Solution Architecture

### Phase 1: Requirement Extraction Pipeline

**New component**: `RequirementExtractor`

```python
@dataclass
class TechnicalRequirements:
    """Domain-specific technical requirements extracted from user request"""
    source_code_modifications: List[str]  # e.g., ["PostgreSQL C code", "pg_duckdb extension"]
    programming_languages: List[str]      # e.g., ["C", "Python"]
    execution_engines: List[str]          # e.g., ["PostgreSQL", "DuckDB"]
    integration_requirements: List[str]   # e.g., ["embed ML model in C", "no external APIs"]
    data_collection_requirements: List[str]  # e.g., ["dual-execution data", "real query features"]
    prohibited_shortcuts: List[str]       # e.g., ["no synthetic data", "no REST APIs", "no simulation"]

class RequirementExtractor:
    """Extract domain-specific technical requirements from research questions"""

    async def extract_requirements(self, research_question: str) -> TechnicalRequirements:
        """Use LLM to extract structured technical requirements"""
        prompt = f"""
        Analyze this research question and extract specific technical requirements:

        {research_question}

        Extract and return in JSON format:
        1. "source_code_modifications": Which codebases must be modified (e.g., "PostgreSQL source code in C")
        2. "programming_languages": Which languages must be used (e.g., "C language for embedding")
        3. "execution_engines": Which engines must be compared or used (e.g., "PostgreSQL vs DuckDB")
        4. "integration_requirements": How components must be integrated (e.g., "embed ML model into database source code")
        5. "data_collection_requirements": What data must be collected (e.g., "dual-execution data on both engines")
        6. "prohibited_shortcuts": What approaches are explicitly forbidden (e.g., "no REST APIs", "no synthetic data", "no simulation")

        Be specific and literal - extract exact technical requirements from the user's wording.
        """
        # LLM call to extract requirements
```

### Phase 2: Constraint-Aware Hypothesis Generation

**Modification**: `HypothesisGenerator.generate_hypotheses()`

Add requirements parameter and inject into prompt:

```python
async def generate_hypotheses(
    self,
    research_question: str,
    technical_requirements: Optional[TechnicalRequirements] = None,  # NEW
    literature_context: Optional[str] = None,
    code_context: Optional[str] = None
) -> List[ResearchHypothesis]:
    """Generate testable hypotheses respecting technical requirements"""

    # Build requirements section
    requirements_text = ""
    if technical_requirements:
        requirements_text = f"""

        CRITICAL TECHNICAL REQUIREMENTS (MUST be incorporated into all hypotheses):

        Source Code Modifications Required:
        {chr(10).join(f"- {req}" for req in technical_requirements.source_code_modifications)}

        Programming Languages Required:
        {chr(10).join(f"- {lang}" for lang in technical_requirements.programming_languages)}

        Execution Engines:
        {chr(10).join(f"- {engine}" for engine in technical_requirements.execution_engines)}

        Integration Requirements:
        {chr(10).join(f"- {req}" for req in technical_requirements.integration_requirements)}

        Data Collection Requirements:
        {chr(10).join(f"- {req}" for req in technical_requirements.data_collection_requirements)}

        PROHIBITED Shortcuts:
        {chr(10).join(f"- ❌ {prohibition}" for prohibition in technical_requirements.prohibited_shortcuts)}

        Your hypotheses MUST test these specific technical requirements, not generic alternatives.
        """

    hypothesis_prompt = f"""
    Based on the research question and context, generate 3-5 testable hypotheses.

    Research Question: {research_question}
    {requirements_text}

    For each hypothesis, provide:
    1. "statement": Clear, testable hypothesis statement that incorporates the technical requirements
    2. "reasoning": Scientific reasoning behind the hypothesis
    3. "testable_predictions": List of specific predictions that can be tested
    4. "success_criteria": Quantitative criteria for validation
    5. "variables": Independent and dependent variables to measure
    6. "technical_constraints": List of technical requirements this hypothesis addresses

    Respond with JSON array of hypothesis objects.
    """
```

### Phase 3: Constraint-Aware Experiment Design

**Modification**: `ExperimentDesigner.design_sequential_experiments()`

Inject technical requirements into experiment design prompt:

```python
async def design_sequential_experiments(
    self,
    hypothesis: ResearchHypothesis,
    num_experiments: int,
    technical_requirements: Optional[TechnicalRequirements] = None,  # NEW
    resources: Optional[Dict[str, Any]] = None,
) -> SequentialExperimentPlan:
    """Design experiments that respect technical requirements"""

    # Build constraints section
    constraints_text = ""
    if technical_requirements:
        constraints_text = f"""

        ═══════════════════════════════════════════════════════════════
        MANDATORY TECHNICAL REQUIREMENTS
        ═══════════════════════════════════════════════════════════════

        Source Code Modifications (YOU MUST MODIFY THESE):
        {chr(10).join(f"✅ MUST modify: {req}" for req in technical_requirements.source_code_modifications)}

        Programming Languages (YOU MUST USE THESE):
        {chr(10).join(f"✅ MUST use: {lang}" for lang in technical_requirements.programming_languages)}

        Execution Engines (YOU MUST RUN EXPERIMENTS ON ALL):
        {chr(10).join(f"✅ MUST execute on: {engine}" for engine in technical_requirements.execution_engines)}

        Integration Requirements (YOU MUST IMPLEMENT):
        {chr(10).join(f"✅ MUST implement: {req}" for req in technical_requirements.integration_requirements)}

        Data Collection (YOU MUST COLLECT THIS DATA):
        {chr(10).join(f"✅ MUST collect: {req}" for req in technical_requirements.data_collection_requirements)}

        ABSOLUTELY PROHIBITED (DO NOT DO THESE):
        {chr(10).join(f"❌ FORBIDDEN: {prohibition}" for prohibition in technical_requirements.prohibited_shortcuts)}

        If you design experiments that violate these requirements, they will be REJECTED.
        """

    base_prompt = f"""
    Design a COMPLETE SEQUENTIAL experimental plan with {num_experiments} experiments to test the following hypothesis.

    Hypothesis: {hypothesis.statement}
    Reasoning: {hypothesis.reasoning}
    Variables: {hypothesis.variables}
    Success Criteria: {hypothesis.success_criteria}
    {constraints_text}

    Design {num_experiments} sequential experiments where:
    - Experiment 1 modifies source code and establishes dual-engine execution baseline
    - Experiment 2 collects dual-execution data and trains ML model
    - Experiment 3 embeds trained model into source code (in required programming language)
    - Experiment 4+ tests the integrated system end-to-end

    CRITICAL: Each experiment MUST explicitly address the technical requirements above.
    """
```

### Phase 4: Pre-Execution Validation

**New component**: `RequirementValidator`

```python
class RequirementValidator:
    """Validate experiment designs against technical requirements"""

    async def validate_experiment_plan(
        self,
        plan: SequentialExperimentPlan,
        requirements: TechnicalRequirements
    ) -> Tuple[bool, List[str]]:
        """Validate experiment plan meets technical requirements"""

        validation_errors = []

        # Check for prohibited shortcuts
        for experiment in plan.experiments:
            methodology_text = str(experiment.methodology).lower()

            for prohibition in requirements.prohibited_shortcuts:
                if any(word in methodology_text for word in prohibition.lower().split()):
                    validation_errors.append(
                        f"Experiment '{experiment.name}' violates prohibition: {prohibition}"
                    )

            # Check for required engines
            engines_mentioned = set()
            for engine in requirements.execution_engines:
                if engine.lower() in methodology_text:
                    engines_mentioned.add(engine)

            if len(engines_mentioned) < len(requirements.execution_engines):
                missing = set(requirements.execution_engines) - engines_mentioned
                validation_errors.append(
                    f"Experiment '{experiment.name}' missing required engines: {missing}"
                )

            # Check for source code modification requirements
            if requirements.source_code_modifications:
                has_modification = any(
                    term in methodology_text
                    for term in ["modify source", "modify code", "edit source", "change source"]
                )
                if not has_modification:
                    validation_errors.append(
                        f"Experiment '{experiment.name}' does not modify source code as required"
                    )

        return len(validation_errors) == 0, validation_errors

    async def validate_execution_results(
        self,
        execution: ExperimentExecution,
        requirements: TechnicalRequirements
    ) -> Tuple[bool, List[str]]:
        """Validate execution results meet technical requirements"""

        validation_errors = []

        # Check final.json for prohibited indicators
        final_data = execution.output_data
        if final_data:
            analysis = final_data.get("analysis", {})
            limitations = final_data.get("limitations", [])

            # Check for prohibited shortcuts in results
            full_text = json.dumps(final_data).lower()
            for prohibition in requirements.prohibited_shortcuts:
                if any(word in full_text for word in prohibition.lower().split()):
                    validation_errors.append(
                        f"Execution used prohibited approach: {prohibition}"
                    )

            # Check for source code modifications
            source_mods = analysis.get("source_code_modifications", [])
            if requirements.source_code_modifications and not source_mods:
                validation_errors.append(
                    "No source code modifications found in results, but required"
                )

        return len(validation_errors) == 0, validation_errors
```

### Phase 5: Technical Guidance Injection

**New component**: `TechnicalGuideProvider`

```python
class TechnicalGuideProvider:
    """Provide specific technical guidance for complex implementation tasks"""

    GUIDES = {
        "embed_sklearn_model_in_c": """
        How to embed a scikit-learn model into PostgreSQL C code:

        1. Export trained model from Python:
           - Use sklearn-porter to convert model to C code
           - Or use m2cgen library to transpile model to C
           - Or serialize model to PMML format and use PMML C parser

        2. Create PostgreSQL C extension:
           - Create extension directory: postgresql/contrib/ml_router/
           - Write C functions that load and use the model
           - Add to PostgreSQL Makefile system

        3. Integrate into query planner:
           - Modify src/backend/optimizer/plan/planner.c
           - Add hook after query parsing, before execution
           - Call your ML model C function to get routing decision

        4. Example code structure:
           ```c
           // contrib/ml_router/ml_router.c
           #include "postgres.h"
           #include "optimizer/planner.h"

           PG_MODULE_MAGIC;

           typedef struct {
               double feature1;
               double feature2;
               // ... extracted features
           } QueryFeatures;

           int predict_engine(QueryFeatures *features) {
               // Model prediction logic (generated by sklearn-porter)
               if (features->feature1 > 0.5) {
                   if (features->feature2 < 0.3) {
                       return ENGINE_POSTGRES;
                   }
               }
               return ENGINE_DUCKDB;
           }
           ```
        """,

        "dual_engine_execution": """
        How to execute queries on both PostgreSQL and DuckDB:

        1. Install pg_duckdb extension:
           - Clone https://github.com/duckdb/pg_duckdb
           - Build and install into PostgreSQL

        2. Configure dual execution:
           - Create DuckDB database file
           - Use pg_duckdb functions to query DuckDB from PostgreSQL

        3. Collect dual-execution data:
           ```python
           import psycopg2

           # Execute on PostgreSQL
           conn_pg = psycopg2.connect("...")
           cursor_pg = conn_pg.cursor()
           cursor_pg.execute("EXPLAIN ANALYZE " + query)
           pg_time = extract_execution_time(cursor_pg.fetchall())

           # Execute on DuckDB via pg_duckdb
           cursor_pg.execute("SELECT duckdb_execute(%s)", (query,))
           duckdb_time = extract_execution_time(cursor_pg.fetchall())

           # Record which was faster
           faster_engine = "duckdb" if duckdb_time < pg_time else "postgres"
           ```

        4. Extract pre-optimization features:
           - Query pg_class for table sizes: SELECT reltuples FROM pg_class
           - Parse query AST for join count
           - Analyze WHERE clause for filter complexity
        """,

        "postgresql_source_modification": """
        How to modify PostgreSQL source code for ML-based routing:

        1. Key files to modify:
           - src/backend/optimizer/plan/planner.c (main planner entry point)
           - src/backend/optimizer/path/allpaths.c (path generation)
           - src/backend/tcop/postgres.c (query processing loop)

        2. Add routing hook:
           ```c
           // In src/backend/optimizer/plan/planner.c
           PlannedStmt *
           planner(Query *parse, const char *query_string, int cursorOptions,
                   ParamListInfo boundParams)
           {
               // Extract features before optimization
               QueryFeatures features = extract_query_features(parse);

               // Call ML model to get routing decision
               int engine = predict_engine(&features);

               if (engine == ENGINE_DUCKDB) {
                   // Route to DuckDB via pg_duckdb
                   return route_to_duckdb(parse, query_string);
               }

               // Continue with normal PostgreSQL planning
               return standard_planner(parse, query_string, cursorOptions, boundParams);
           }
           ```

        3. Build and test:
           - cd postgresql && make && make install
           - initdb -D data
           - pg_ctl -D data start
           - psql -d postgres -c "SELECT test_ml_routing();"
        """
    }

    def get_guide(self, task: str) -> str:
        """Get technical guidance for a specific task"""
        return self.GUIDES.get(task, "No specific guidance available")

    def inject_guides_into_prompt(
        self,
        base_prompt: str,
        requirements: TechnicalRequirements
    ) -> str:
        """Inject relevant technical guides into experiment prompt"""

        guides_section = "\n\n═══════════════════════════════════════════════════════════════\n"
        guides_section += "TECHNICAL IMPLEMENTATION GUIDANCE\n"
        guides_section += "═══════════════════════════════════════════════════════════════\n\n"

        # Determine which guides to include
        if any("embed" in req.lower() and "model" in req.lower()
               for req in requirements.integration_requirements):
            guides_section += self.get_guide("embed_sklearn_model_in_c") + "\n\n"

        if len(requirements.execution_engines) > 1:
            guides_section += self.get_guide("dual_engine_execution") + "\n\n"

        if any("postgresql" in mod.lower() or "postgres" in mod.lower()
               for mod in requirements.source_code_modifications):
            guides_section += self.get_guide("postgresql_source_modification") + "\n\n"

        # Insert guides before "FINAL DELIVERABLE" section
        if "FINAL DELIVERABLE" in base_prompt:
            return base_prompt.replace("FINAL DELIVERABLE", guides_section + "FINAL DELIVERABLE")
        else:
            return base_prompt + "\n\n" + guides_section
```

## Implementation Plan

### Files to Modify

1. **backend/app/core/research_engines/scientific_research.py**
   - Add `RequirementExtractor` class (lines 220-300)
   - Add `RequirementValidator` class (lines 300-400)
   - Add `TechnicalGuideProvider` class (lines 400-600)
   - Modify `HypothesisGenerator.generate_hypotheses()` (lines 251-384)
   - Modify `ExperimentDesigner.design_sequential_experiments()` (lines 658-827)
   - Modify `ExperimentExecutor.execute_sequential_experiments()` (lines 1305-1492)
   - Add pre-execution validation before OpenHands call
   - Add post-execution validation with retry on failure

2. **backend/app/routers/smart_router.py**
   - Modify `execute_scientific()` function (lines 303-400)
   - Add requirement extraction before calling `conduct_research()`
   - Pass requirements through to engine

3. **backend/app/core/research_engines/scientific_research.py** (ScientificResearchEngine)
   - Modify `conduct_research()` (lines 3658-3737)
   - Add `technical_requirements` parameter
   - Pass requirements to hypothesis generation and experiment design

### Testing Strategy

1. **Unit tests**:
   - `test_requirement_extractor.py`: Test extraction from various user queries
   - `test_requirement_validator.py`: Test validation logic
   - `test_technical_guide_provider.py`: Test guide injection

2. **Integration tests**:
   - `test_constrained_hypothesis_generation.py`: Verify requirements propagate to hypotheses
   - `test_constrained_experiment_design.py`: Verify experiment designs respect constraints
   - `test_experiment_validation.py`: Verify validation catches violations

3. **End-to-end tests**:
   - Run user's actual query: "modify postgres and pg_duckdb source code..."
   - Verify experiments collect dual-execution data
   - Verify models are embedded in C code
   - Verify no REST APIs or synthetic data used

## Expected Outcomes After Implementation

1. ✅ Hypotheses will explicitly test dual-engine execution (PostgreSQL vs DuckDB)
2. ✅ Experiment designs will require source code modification in C
3. ✅ Experiment designs will explicitly prohibit REST APIs and synthetic data
4. ✅ Technical guides will help OpenHands implement complex tasks correctly
5. ✅ Pre-execution validation will reject non-compliant experiment plans
6. ✅ Post-execution validation will detect and retry experiments that took shortcuts
7. ✅ Final results will contain real dual-execution data and embedded ML models

## Conclusion

The root cause of experiment failures was **loss of technical requirements during hypothesis generation and experiment design**. The generic prompts allowed LLMs to design experiments that technically "test the hypothesis" but violate the user's specific implementation requirements.

The solution requires **requirement extraction, constraint injection, validation loops, and technical guidance** at every stage of the research pipeline to ensure experiments follow the user's exact specifications rather than generic alternatives.
