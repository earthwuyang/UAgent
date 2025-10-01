# Implementation Plan: Fix Scientific Experiment Requirement Compliance

## Overview

This document provides step-by-step instructions for implementing the fixes identified in `EXPERIMENT_FAILURE_ANALYSIS.md`. The fixes ensure that scientific experiments respect the user's technical requirements rather than taking generic shortcuts.

## Phase 1: Add RequirementExtractor Component

### Step 1.1: Create TechnicalRequirements dataclass

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: After line 209 (after ResearchIdea dataclass)

```python
@dataclass
class TechnicalRequirements:
    """Domain-specific technical requirements extracted from user request"""
    source_code_modifications: List[str] = field(default_factory=list)
    programming_languages: List[str] = field(default_factory=list)
    execution_engines: List[str] = field(default_factory=list)
    integration_requirements: List[str] = field(default_factory=list)
    data_collection_requirements: List[str] = field(default_factory=list)
    prohibited_shortcuts: List[str] = field(default_factory=list)
    technical_guidance_needed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_code_modifications": self.source_code_modifications,
            "programming_languages": self.programming_languages,
            "execution_engines": self.execution_engines,
            "integration_requirements": self.integration_requirements,
            "data_collection_requirements": self.data_collection_requirements,
            "prohibited_shortcuts": self.prohibited_shortcuts,
            "technical_guidance_needed": self.technical_guidance_needed,
        }
```

### Step 1.2: Create RequirementExtractor class

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: After line 220 (before HypothesisGenerator class)

```python
class RequirementExtractor:
    """Extract domain-specific technical requirements from research questions"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.max_generation_tokens = get_max_tokens_from_env()

    async def extract_requirements(
        self,
        research_question: str
    ) -> TechnicalRequirements:
        """Extract structured technical requirements from research question"""

        extraction_prompt = f"""
Analyze this research question and extract SPECIFIC technical requirements:

{research_question}

Extract and return in JSON format with these exact keys:
1. "source_code_modifications": Which codebases/files must be modified (e.g., ["PostgreSQL source code", "pg_duckdb extension"])
2. "programming_languages": Which languages must be used (e.g., ["C language", "Python"])
3. "execution_engines": Which engines must be compared or used (e.g., ["PostgreSQL", "DuckDB"])
4. "integration_requirements": How components must be integrated (e.g., ["embed ML model into database source code", "no external APIs"])
5. "data_collection_requirements": What data must be collected (e.g., ["dual-execution data on both engines", "real query execution times"])
6. "prohibited_shortcuts": What approaches are explicitly forbidden (e.g., ["no REST APIs", "no synthetic data", "no simulation", "no mock implementations"])
7. "technical_guidance_needed": What complex tasks need implementation guidance (e.g., ["embed sklearn model in C", "modify PostgreSQL planner", "dual-engine execution"])

Be SPECIFIC and LITERAL - extract exact technical requirements from the user's wording.
If the user says "modify postgres source code", include that exact phrase.
If the user says "embed model in C language", include that exact requirement.

Return ONLY a valid JSON object with these keys.
"""

        response = await self.llm_client.generate(
            extraction_prompt,
            max_tokens=self.max_generation_tokens,
            temperature=0.1  # Low temperature for consistency
        )

        try:
            sanitized = sanitize_json_strings(str(response))
            req_dict = safe_json_loads(sanitized)
        except JsonParseError as exc:
            self.logger.error(f"Failed to parse requirements: {exc}")
            # Return empty requirements on parse failure
            req_dict = {}

        return TechnicalRequirements(
            source_code_modifications=req_dict.get("source_code_modifications", []),
            programming_languages=req_dict.get("programming_languages", []),
            execution_engines=req_dict.get("execution_engines", []),
            integration_requirements=req_dict.get("integration_requirements", []),
            data_collection_requirements=req_dict.get("data_collection_requirements", []),
            prohibited_shortcuts=req_dict.get("prohibited_shortcuts", []),
            technical_guidance_needed=req_dict.get("technical_guidance_needed", []),
        )
```

## Phase 2: Add RequirementValidator Component

### Step 2.1: Create RequirementValidator class

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: After RequirementExtractor class

```python
class RequirementValidator:
    """Validate experiment designs and results against technical requirements"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def validate_experiment_plan(
        self,
        plan: SequentialExperimentPlan,
        requirements: TechnicalRequirements
    ) -> Tuple[bool, List[str]]:
        """Validate experiment plan meets technical requirements before execution"""

        validation_errors = []

        # Validate each experiment in the plan
        for exp_idx, experiment in enumerate(plan.experiments, start=1):
            methodology_text = str(experiment.methodology).lower()
            description_text = str(experiment.description).lower()
            full_text = methodology_text + " " + description_text

            # Check for prohibited shortcuts
            for prohibition in requirements.prohibited_shortcuts:
                prohibition_words = prohibition.lower().split()
                # Check if ANY key words from prohibition appear in methodology
                if any(word in full_text for word in prohibition_words if len(word) > 3):
                    validation_errors.append(
                        f"❌ Experiment {exp_idx} '{experiment.name}' may violate prohibition: {prohibition}"
                    )

            # Check for required execution engines
            if len(requirements.execution_engines) > 1:
                engines_mentioned = set()
                for engine in requirements.execution_engines:
                    if engine.lower() in full_text:
                        engines_mentioned.add(engine)

                if len(engines_mentioned) < len(requirements.execution_engines):
                    missing = set(requirements.execution_engines) - engines_mentioned
                    validation_errors.append(
                        f"❌ Experiment {exp_idx} '{experiment.name}' missing required engines: {missing}"
                    )

            # Check for source code modification requirements
            if requirements.source_code_modifications:
                modification_indicators = [
                    "modify source", "modify code", "edit source", "change source",
                    "modify postgres", "modify pg_duckdb", "edit c code",
                    "change postgres", "patch source"
                ]
                has_modification = any(indicator in full_text for indicator in modification_indicators)

                if not has_modification:
                    validation_errors.append(
                        f"❌ Experiment {exp_idx} '{experiment.name}' does not modify source code as required: {requirements.source_code_modifications}"
                    )

            # Check for programming language requirements
            if requirements.programming_languages:
                for lang in requirements.programming_languages:
                    if lang.lower() not in full_text:
                        validation_errors.append(
                            f"⚠️  Experiment {exp_idx} '{experiment.name}' does not mention required language: {lang}"
                        )

        # Validate shared setup
        shared_setup_text = str(plan.shared_setup).lower()
        for prohibition in requirements.prohibited_shortcuts:
            prohibition_words = prohibition.lower().split()
            if any(word in shared_setup_text for word in prohibition_words if len(word) > 3):
                validation_errors.append(
                    f"❌ Shared setup violates prohibition: {prohibition}"
                )

        is_valid = len(validation_errors) == 0
        if not is_valid:
            self.logger.warning(f"Experiment plan validation failed with {len(validation_errors)} errors")
            for error in validation_errors:
                self.logger.warning(error)

        return is_valid, validation_errors

    async def validate_execution_results(
        self,
        execution: ExperimentExecution,
        requirements: TechnicalRequirements
    ) -> Tuple[bool, List[str]]:
        """Validate execution results meet technical requirements after execution"""

        validation_errors = []

        # Check output_data for compliance
        final_data = execution.output_data
        if not final_data:
            validation_errors.append("❌ No output data found in execution results")
            return False, validation_errors

        # Check for prohibited shortcuts in results
        full_text = json.dumps(final_data).lower()
        for prohibition in requirements.prohibited_shortcuts:
            prohibition_words = prohibition.lower().split()
            if any(word in full_text for word in prohibition_words if len(word) > 3):
                validation_errors.append(
                    f"❌ Execution results contain prohibited approach: {prohibition}"
                )

        # Check analysis section for source code modifications
        analysis = final_data.get("analysis", {})
        source_mods = analysis.get("source_code_modifications", [])

        if requirements.source_code_modifications and not source_mods:
            validation_errors.append(
                f"❌ No source code modifications found in results, but required: {requirements.source_code_modifications}"
            )

        # Check for dual-engine execution if required
        if len(requirements.execution_engines) > 1:
            engines_in_results = set()
            for engine in requirements.execution_engines:
                if engine.lower() in full_text:
                    engines_in_results.add(engine)

            if len(engines_in_results) < len(requirements.execution_engines):
                missing = set(requirements.execution_engines) - engines_in_results
                validation_errors.append(
                    f"❌ Results missing data from required engines: {missing}"
                )

        is_valid = len(validation_errors) == 0
        if not is_valid:
            self.logger.warning(f"Execution validation failed with {len(validation_errors)} errors")
            for error in validation_errors:
                self.logger.warning(error)

        return is_valid, validation_errors
```

## Phase 3: Add TechnicalGuideProvider Component

### Step 3.1: Create TechnicalGuideProvider class

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: After RequirementValidator class

```python
class TechnicalGuideProvider:
    """Provide specific technical guidance for complex implementation tasks"""

    GUIDES = {
        "embed sklearn model in c": """
═══════════════════════════════════════════════════════════════
TECHNICAL GUIDE: Embedding scikit-learn ML Models in PostgreSQL C Code
═══════════════════════════════════════════════════════════════

Option 1: Use sklearn-porter (Recommended for simple models)
-------------------------------------------------------------
1. Install: pip install sklearn-porter
2. Export model:
   from sklearn_porter import Porter
   porter = Porter(model, language='c')
   output = porter.export()
   # Saves model as C code

Option 2: Use m2cgen (Supports more model types)
-------------------------------------------------------------
1. Install: pip install m2cgen
2. Export model:
   import m2cgen as m2c
   code = m2c.export_to_c(model)
   # Generates C prediction function

Option 3: Manual serialization with PMML
-------------------------------------------------------------
1. Export to PMML: sklearn2pmml
2. Use C PMML parser library
3. Load model at PostgreSQL startup

PostgreSQL C Extension Integration:
-------------------------------------------------------------
Create extension directory: postgresql/contrib/ml_router/

File: contrib/ml_router/ml_router.c
```c
#include "postgres.h"
#include "fmgr.h"
#include "optimizer/planner.h"
#include "nodes/nodes.h"

PG_MODULE_MAGIC;

// Model prediction function (generated by sklearn-porter/m2cgen)
static double predict_score(double *features, int n_features) {
    // Generated decision tree logic
    if (features[0] > 0.5) {
        if (features[1] < 0.3) {
            return 1.0;  // Route to engine 1
        }
        return 0.0;  // Route to engine 2
    }
    return 0.0;
}

// PostgreSQL C function callable from SQL
PG_FUNCTION_INFO_V1(ml_route_query);
Datum
ml_route_query(PG_FUNCTION_ARGS)
{
    // Extract features from current query
    double features[10];
    // ... feature extraction logic ...

    double score = predict_score(features, 10);
    PG_RETURN_FLOAT8(score);
}
```

File: contrib/ml_router/Makefile
```makefile
MODULES = ml_router
EXTENSION = ml_router
DATA = ml_router--1.0.sql

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/ml_router
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif
```

Build and install:
```bash
cd postgresql/contrib/ml_router
make
make install
```

Test:
```sql
CREATE EXTENSION ml_router;
SELECT ml_route_query();
```
""",

        "dual-engine execution": """
═══════════════════════════════════════════════════════════════
TECHNICAL GUIDE: Dual-Engine Query Execution (PostgreSQL + DuckDB)
═══════════════════════════════════════════════════════════════

Method 1: Use pg_duckdb Extension (Recommended)
-------------------------------------------------------------
1. Clone and build pg_duckdb:
   git clone https://github.com/duckdb/pg_duckdb
   cd pg_duckdb
   make
   make install

2. Enable in PostgreSQL:
   CREATE EXTENSION pg_duckdb;

3. Execute on both engines:
   Python script:
   ```python
   import psycopg2
   import time

   conn = psycopg2.connect("dbname=test user=postgres")
   cursor = conn.cursor()

   query = "SELECT COUNT(*) FROM large_table WHERE value > 100"

   # Execute on PostgreSQL
   start = time.time()
   cursor.execute(f"EXPLAIN ANALYZE {query}")
   pg_plan = cursor.fetchall()
   cursor.execute(query)
   pg_result = cursor.fetchall()
   pg_time = time.time() - start

   # Execute on DuckDB via pg_duckdb
   start = time.time()
   cursor.execute(f"SELECT duckdb.execute(%s)", (query,))
   duckdb_result = cursor.fetchall()
   duckdb_time = time.time() - start

   # Record training data
   faster_engine = "duckdb" if duckdb_time < pg_time else "postgres"
   features = extract_features(query, pg_plan)
   training_data.append((features, faster_engine, pg_time, duckdb_time))
   ```

Method 2: Separate Connection Pools
-------------------------------------------------------------
1. Install DuckDB Python: pip install duckdb
2. Maintain separate connections:
   ```python
   import psycopg2
   import duckdb

   pg_conn = psycopg2.connect("...")
   duck_conn = duckdb.connect("database.duckdb")

   # Execute on both
   pg_cursor = pg_conn.cursor()
   pg_cursor.execute(query)
   pg_result = pg_cursor.fetchall()

   duck_cursor = duck_conn.cursor()
   duck_cursor.execute(query)
   duck_result = duck_cursor.fetchall()
   ```

Feature Extraction from PostgreSQL Kernel:
-------------------------------------------------------------
Extract BEFORE query optimization:

```python
def extract_preopt_features(query, pg_cursor):
    # Get query plan
    pg_cursor.execute(f"EXPLAIN (FORMAT JSON, VERBOSE) {query}")
    plan = pg_cursor.fetchone()[0][0]

    features = {}

    # Feature 1: Table sizes from pg_class
    tables = extract_tables_from_plan(plan)
    for table in tables:
        pg_cursor.execute(f"SELECT reltuples, relpages FROM pg_class WHERE relname='{table}'")
        row_count, page_count = pg_cursor.fetchone()
        features[f"{table}_rows"] = row_count
        features[f"{table}_pages"] = page_count

    # Feature 2: Join count
    features["join_count"] = count_joins_in_plan(plan)

    # Feature 3: Filter complexity
    features["filter_count"] = count_filters_in_plan(plan)

    # Feature 4: Aggregation count
    features["agg_count"] = count_aggregations_in_plan(plan)

    return features
```

Data Collection Loop:
-------------------------------------------------------------
```python
training_data = []
for query in tpch_queries:
    # Extract features
    features = extract_preopt_features(query, pg_cursor)

    # Execute on both engines
    pg_time = execute_on_postgres(query, pg_cursor)
    duck_time = execute_on_duckdb(query, duck_cursor)

    # Label with faster engine
    label = "postgres" if pg_time < duck_time else "duckdb"

    training_data.append({
        "features": features,
        "label": label,
        "pg_time": pg_time,
        "duck_time": duck_time,
    })

# Train model
X = [d["features"] for d in training_data]
y = [d["label"] for d in training_data]
model = RandomForestClassifier()
model.fit(X, y)
```
""",

        "modify postgresql planner": """
═══════════════════════════════════════════════════════════════
TECHNICAL GUIDE: Modifying PostgreSQL Query Planner for ML Routing
═══════════════════════════════════════════════════════════════

Key Files to Modify:
-------------------------------------------------------------
1. src/backend/optimizer/plan/planner.c - Main planner entry
2. src/backend/tcop/postgres.c - Query execution loop
3. src/include/optimizer/planner.h - Planner headers

Step 1: Add Hook in Planner
-------------------------------------------------------------
File: src/backend/optimizer/plan/planner.c

Find the planner() function (around line 280):

```c
PlannedStmt *
planner(Query *parse, const char *query_string, int cursorOptions,
        ParamListInfo boundParams)
{
    PlannedStmt *result;
    PlannerGlobal *glob;
    // ... existing code ...

    // ADD ML ROUTING HOOK HERE (before standard_planner call)
    if (ml_routing_enabled)
    {
        MLRoutingDecision decision = ml_route_query(parse, query_string);
        if (decision.route_to_duckdb)
        {
            // Route to DuckDB via pg_duckdb
            result = duckdb_execute_query(parse, query_string);
            return result;
        }
    }

    // Continue with standard PostgreSQL planning
    result = standard_planner(parse, query_string, cursorOptions, boundParams);
    return result;
}
```

Step 2: Implement ML Routing Function
-------------------------------------------------------------
File: src/backend/optimizer/plan/ml_router.c (NEW FILE)

```c
#include "postgres.h"
#include "optimizer/planner.h"
#include "parser/parsetree.h"
#include "catalog/pg_class.h"

typedef struct MLRoutingDecision {
    bool route_to_duckdb;
    double confidence;
} MLRoutingDecision;

// Extract features from Query structure
static double* extract_query_features(Query *parse) {
    double *features = palloc(sizeof(double) * 10);

    // Feature 1: Number of relations
    features[0] = (double)list_length(parse->rtable);

    // Feature 2: Number of joins
    features[1] = (double)count_joins(parse);

    // Feature 3: Aggregation presence
    features[2] = parse->hasAggs ? 1.0 : 0.0;

    // Feature 4: Subquery presence
    features[3] = parse->hasSubLinks ? 1.0 : 0.0;

    // Feature 5-10: Table sizes from pg_class
    // ... query pg_class for reltuples ...

    return features;
}

// ML model prediction (generated from sklearn)
static double predict_duckdb_score(double *features) {
    // Decision tree logic generated by sklearn-porter
    if (features[0] > 2.5) {
        if (features[1] > 0.5) {
            return 0.8;  // High confidence for DuckDB
        }
        return 0.3;
    }
    return 0.1;  // Low confidence, use PostgreSQL
}

// Main routing function
MLRoutingDecision ml_route_query(Query *parse, const char *query_string) {
    MLRoutingDecision decision;
    double *features = extract_query_features(parse);
    double score = predict_duckdb_score(features);

    decision.route_to_duckdb = (score > 0.5);
    decision.confidence = score;

    pfree(features);
    return decision;
}
```

Step 3: Add to PostgreSQL Build System
-------------------------------------------------------------
File: src/backend/optimizer/plan/Makefile

Add ml_router.c to OBJS:
```makefile
OBJS = planner.o planmain.o createplan.o ml_router.o ...
```

Step 4: Rebuild and Test
-------------------------------------------------------------
```bash
cd postgresql
make clean
make -j$(nproc)
make install

# Initialize new database
initdb -D data_test
pg_ctl -D data_test start

# Test ML routing
psql -d postgres -c "SET ml_routing_enabled = on;"
psql -d postgres -c "SELECT * FROM test_table WHERE id > 100;"
# Check logs for ML routing decision
```

Step 5: Add Configuration Parameters
-------------------------------------------------------------
File: src/backend/utils/misc/guc.c

```c
static bool ml_routing_enabled = false;

{
    {"ml_routing_enabled", PGC_USERSET, QUERY_TUNING_METHOD,
        gettext_noop("Enables ML-based query routing to DuckDB."),
        NULL
    },
    &ml_routing_enabled,
    false,
    NULL, NULL, NULL
},
```

Now users can enable/disable:
```sql
SET ml_routing_enabled = on;
```
"""
    }

    def get_guide(self, task_key: str) -> str:
        """Get technical guidance for a specific task"""
        # Normalize task key
        task_key_lower = task_key.lower().strip()

        # Check for exact matches
        if task_key_lower in self.GUIDES:
            return self.GUIDES[task_key_lower]

        # Check for partial matches
        for guide_key, guide_content in self.GUIDES.items():
            if task_key_lower in guide_key or guide_key in task_key_lower:
                return guide_content

        return f"# Technical Guide Requested: {task_key}\n\n(No specific guide available - implement using standard practices)"

    def inject_guides_into_prompt(
        self,
        base_prompt: str,
        requirements: TechnicalRequirements
    ) -> str:
        """Inject relevant technical guides into experiment prompt"""

        if not requirements.technical_guidance_needed:
            return base_prompt

        guides_section = "\n\n═══════════════════════════════════════════════════════════════\n"
        guides_section += "TECHNICAL IMPLEMENTATION GUIDANCE (HOW TO ACCOMPLISH TASKS)\n"
        guides_section += "═══════════════════════════════════════════════════════════════\n"
        guides_section += "\nThe following guides provide SPECIFIC implementation details for complex tasks.\n"
        guides_section += "Follow these guides EXACTLY to accomplish the technical requirements.\n\n"

        # Include all requested guides
        for guidance_task in requirements.technical_guidance_needed:
            guide_content = self.get_guide(guidance_task)
            guides_section += guide_content + "\n\n"

        # Insert guides before "FINAL DELIVERABLE" section
        if "FINAL DELIVERABLE" in base_prompt:
            return base_prompt.replace(
                "═══════════════════════════════════════════════════════════════\nFINAL DELIVERABLE",
                guides_section + "═══════════════════════════════════════════════════════════════\nFINAL DELIVERABLE"
            )
        else:
            return base_prompt + "\n\n" + guides_section
```

## Phase 4: Integrate Components into Research Pipeline

### Step 4.1: Modify ScientificResearchEngine.__init__

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `ScientificResearchEngine.__init__` method (around line 2253)

Add after line 2254:

```python
self.hypothesis_generator = HypothesisGenerator(llm_client)
self.experiment_designer = ExperimentDesigner(llm_client)
# ADD THESE:
self.requirement_extractor = RequirementExtractor(llm_client)
self.requirement_validator = RequirementValidator()
self.technical_guide_provider = TechnicalGuideProvider()
```

### Step 4.2: Modify conduct_research method

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: `conduct_research` method (around line 3658)

**Current signature**:
```python
async def conduct_research(
    self,
    research_question: str,
    include_literature_review: bool = True,
    include_code_analysis: bool = True,
    enable_iteration: bool = True,
    session_id: Optional[str] = None
) -> ScientificResearchResult:
```

**Change to**:
```python
async def conduct_research(
    self,
    research_question: str,
    include_literature_review: bool = True,
    include_code_analysis: bool = True,
    enable_iteration: bool = True,
    session_id: Optional[str] = None,
    technical_requirements: Optional[TechnicalRequirements] = None  # NEW
) -> ScientificResearchResult:
```

**Add after line 3716** (after initializing progress):

```python
# Extract technical requirements if not provided
if technical_requirements is None:
    await self._log_progress(
        session_id,
        phase="requirement_extraction",
        progress=7.0,
        message="Extracting technical requirements from research question",
        metadata={"parent_id": root_id}
    )
    technical_requirements = await self.requirement_extractor.extract_requirements(research_question)
    self.logger.info(f"Extracted technical requirements: {technical_requirements.to_dict()}")

# Store requirements in result for later access
result.technical_requirements = technical_requirements
```

### Step 4.3: Add technical_requirements to ScientificResearchResult dataclass

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: `ScientificResearchResult` dataclass (around line 188)

Add field:
```python
@dataclass
class ScientificResearchResult:
    # ... existing fields ...
    recommendations: List[str] = field(default_factory=list)
    debates: List[Debate] = field(default_factory=list)
    technical_requirements: Optional[TechnicalRequirements] = None  # NEW
```

## Phase 5: Integrate Requirements into Hypothesis Generation

### Step 5.1: Modify HypothesisGenerator.generate_hypotheses

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: `generate_hypotheses` method (line 251)

**Current signature**:
```python
async def generate_hypotheses(
    self,
    research_question: str,
    literature_context: Optional[str] = None,
    code_context: Optional[str] = None
) -> List[ResearchHypothesis]:
```

**Change to**:
```python
async def generate_hypotheses(
    self,
    research_question: str,
    technical_requirements: Optional[TechnicalRequirements] = None,  # NEW
    literature_context: Optional[str] = None,
    code_context: Optional[str] = None
) -> List[ResearchHypothesis]:
```

**Modify prompt building** (around line 265):

```python
context = f"Research Question: {research_question}\n"
if literature_context:
    context += f"Literature Context: {literature_context[:1000]}...\n"
if code_context:
    context += f"Code Context: {code_context[:1000]}...\n"

# ADD TECHNICAL REQUIREMENTS SECTION
requirements_text = ""
if technical_requirements:
    requirements_text = f"""

═══════════════════════════════════════════════════════════════
CRITICAL TECHNICAL REQUIREMENTS (MUST BE INCORPORATED)
═══════════════════════════════════════════════════════════════

Your hypotheses MUST test these SPECIFIC technical requirements:

Source Code Modifications Required:
{chr(10).join(f"  ✅ MUST modify: {req}" for req in technical_requirements.source_code_modifications)}

Programming Languages Required:
{chr(10).join(f"  ✅ MUST use: {lang}" for lang in technical_requirements.programming_languages)}

Execution Engines:
{chr(10).join(f"  ✅ MUST execute on: {engine}" for engine in technical_requirements.execution_engines)}

Integration Requirements:
{chr(10).join(f"  ✅ MUST implement: {req}" for req in technical_requirements.integration_requirements)}

Data Collection Requirements:
{chr(10).join(f"  ✅ MUST collect: {req}" for req in technical_requirements.data_collection_requirements)}

ABSOLUTELY PROHIBITED Shortcuts:
{chr(10).join(f"  ❌ FORBIDDEN: {prohibition}" for prohibition in technical_requirements.prohibited_shortcuts)}

Your hypotheses MUST test these specific technical requirements.
Do NOT generate generic hypotheses that avoid these requirements.
"""

hypothesis_prompt = f"""
Based on the research question and context, generate 3-5 testable hypotheses.

{context}
{requirements_text}

For each hypothesis, provide:
1. "statement": Clear, testable hypothesis statement that directly tests the technical requirements above
2. "reasoning": Scientific reasoning behind the hypothesis
3. "testable_predictions": List of specific predictions that can be tested
4. "success_criteria": Quantitative criteria for validation
5. "variables": Independent and dependent variables to measure

IMPORTANT: Each hypothesis MUST incorporate and test the technical requirements specified above.

Respond with JSON array of hypothesis objects.
"""
```

### Step 5.2: Update conduct_research to pass requirements

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `conduct_research` method, where hypotheses are generated (around line 2831)

Find the line:
```python
hypotheses = await self.hypothesis_generator.generate_hypotheses(
    research_question,
    literature_context=literature_context,
    code_context=code_context
)
```

**Change to**:
```python
hypotheses = await self.hypothesis_generator.generate_hypotheses(
    research_question,
    technical_requirements=technical_requirements,  # NEW
    literature_context=literature_context,
    code_context=code_context
)
```

## Phase 6: Integrate Requirements into Experiment Design

### Step 6.1: Modify ExperimentDesigner.design_sequential_experiments

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: `design_sequential_experiments` method (line 658)

**Current signature**:
```python
async def design_sequential_experiments(
    self,
    hypothesis: ResearchHypothesis,
    num_experiments: int,
    resources: Optional[Dict[str, Any]] = None,
) -> SequentialExperimentPlan:
```

**Change to**:
```python
async def design_sequential_experiments(
    self,
    hypothesis: ResearchHypothesis,
    num_experiments: int,
    technical_requirements: Optional[TechnicalRequirements] = None,  # NEW
    technical_guide_provider: Optional[TechnicalGuideProvider] = None,  # NEW
    resources: Optional[Dict[str, Any]] = None,
) -> SequentialExperimentPlan:
```

**Modify prompt building** (around line 672):

Add before line 710 (before "Respond with a raw JSON object"):

```python
# Build technical requirements constraints
constraints_text = ""
if technical_requirements:
    constraints_text = f"""

═══════════════════════════════════════════════════════════════
MANDATORY TECHNICAL REQUIREMENTS
═══════════════════════════════════════════════════════════════

ALL experiments MUST comply with these requirements:

Source Code Modifications (YOU MUST MODIFY THESE IN YOUR EXPERIMENTS):
{chr(10).join(f"  ✅ REQUIRED: {req}" for req in technical_requirements.source_code_modifications)}

Programming Languages (YOU MUST USE THESE):
{chr(10).join(f"  ✅ REQUIRED: {lang}" for lang in technical_requirements.programming_languages)}

Execution Engines (YOU MUST RUN EXPERIMENTS ON ALL OF THESE):
{chr(10).join(f"  ✅ REQUIRED: {engine}" for engine in technical_requirements.execution_engines)}

Integration Requirements (YOU MUST IMPLEMENT THESE):
{chr(10).join(f"  ✅ REQUIRED: {req}" for req in technical_requirements.integration_requirements)}

Data Collection (YOU MUST COLLECT THIS SPECIFIC DATA):
{chr(10).join(f"  ✅ REQUIRED: {req}" for req in technical_requirements.data_collection_requirements)}

ABSOLUTELY PROHIBITED (YOU MUST NOT DO THESE):
{chr(10).join(f"  ❌ FORBIDDEN: {prohibition}" for prohibition in technical_requirements.prohibited_shortcuts)}

CRITICAL: If your experiment designs violate any of these requirements, they will be REJECTED.
For example:
- DO NOT use REST APIs if "no REST APIs" is in prohibited shortcuts
- DO NOT use synthetic data if "no synthetic data" is in prohibited shortcuts
- DO NOT skip source code modifications if source code modifications are required
"""

# Update base_prompt to include constraints
base_prompt = f"""
Design a COMPLETE SEQUENTIAL experimental plan with {num_experiments} experiments to test the following hypothesis.
These experiments will be executed ONE AFTER ANOTHER in the SAME computational environment,
so later experiments CAN and SHOULD build upon the code, data, and results from earlier experiments.

Hypothesis: {hypothesis.statement}
Reasoning: {hypothesis.reasoning}
Variables: {hypothesis.variables}
Success Criteria: {hypothesis.success_criteria}
Available Resources: {resources or "Standard computational resources"}
{constraints_text}

Design {num_experiments} sequential experiments where:
- Experiment 1 modifies required source code and establishes baseline with required engines
- Experiment 2 collects required data types from all required engines
- Experiment 3 trains ML model and embeds it in required programming language
- Experiment 4+ tests the integrated system end-to-end

CRITICAL RULES:
1. Each experiment MUST explicitly address the technical requirements above
2. Use EXACT technologies specified (e.g., if "PostgreSQL vs DuckDB" required, must use both)
3. Do NOT use shortcuts that violate prohibited approaches
4. Build on previous experiments - reuse installations and code from earlier steps

Provide the sequential plan in JSON format with:
[... rest of prompt ...]
"""

# Inject technical guides if available
if technical_guide_provider and technical_requirements:
    base_prompt = technical_guide_provider.inject_guides_into_prompt(
        base_prompt,
        technical_requirements
    )
```

### Step 6.2: Update conduct_research to pass requirements to experiment design

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `_test_research_idea` method, where sequential plan is designed (around line 3017)

Find the line:
```python
sequential_plan = await self.experiment_designer.design_sequential_experiments(
    hypothesis=hypothesis,
    num_experiments=self.experiments_per_hypothesis,
)
```

**Change to**:
```python
sequential_plan = await self.experiment_designer.design_sequential_experiments(
    hypothesis=hypothesis,
    num_experiments=self.experiments_per_hypothesis,
    technical_requirements=result.technical_requirements,  # NEW
    technical_guide_provider=self.technical_guide_provider,  # NEW
)
```

## Phase 7: Add Validation Loop

### Step 7.1: Add pre-execution validation

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `_test_research_idea` method, AFTER sequential plan is designed (around line 3022)

Add:

```python
sequential_plan = await self.experiment_designer.design_sequential_experiments(
    hypothesis=hypothesis,
    num_experiments=self.experiments_per_hypothesis,
    technical_requirements=result.technical_requirements,
    technical_guide_provider=self.technical_guide_provider,
)

# VALIDATE PLAN BEFORE EXECUTION
if result.technical_requirements:
    is_valid, validation_errors = await self.requirement_validator.validate_experiment_plan(
        sequential_plan,
        result.technical_requirements
    )

    if not is_valid:
        self.logger.warning(f"Experiment plan validation failed with {len(validation_errors)} errors")
        for error in validation_errors:
            self.logger.warning(f"  - {error}")

        # Log validation failure
        if session_id and idea.node_id:
            await self._log_progress(
                session_id,
                phase=f"{idea.id}_validation_failed",
                progress=iteration_progress + per_iteration_increment * 0.1,
                message=f"Experiment plan validation failed - redesigning with stricter constraints",
                metadata={
                    "parent_id": idea.node_id,
                    "node_type": "error",
                    "validation_errors": validation_errors[:5],  # First 5 errors
                },
                parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
            )

        # Retry with validation errors as feedback
        self.logger.info("Retrying experiment design with validation feedback")
        sequential_plan = await self.experiment_designer.design_sequential_experiments(
            hypothesis=hypothesis,
            num_experiments=self.experiments_per_hypothesis,
            technical_requirements=result.technical_requirements,
            technical_guide_provider=self.technical_guide_provider,
            resources={"validation_feedback": validation_errors},  # Pass errors as feedback
        )

        # Validate again
        is_valid, validation_errors = await self.requirement_validator.validate_experiment_plan(
            sequential_plan,
            result.technical_requirements
        )

        if not is_valid:
            self.logger.error("Experiment plan still invalid after redesign - proceeding with warnings")
```

### Step 7.2: Add post-execution validation

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `_execute_sequential_plan_with_retries` method, AFTER execution completes

Find where executions are returned (around line 1490) and add validation:

```python
# Parse results for each experiment
executions = self._parse_comprehensive_results(
    comprehensive_execution,
    plan,
    workspace_id=workspace_id,
    session_id=session_id
)

# VALIDATE EXECUTION RESULTS
if hasattr(self, 'requirement_validator') and session_context.technical_requirements:
    for execution in executions:
        is_valid, validation_errors = await self.requirement_validator.validate_execution_results(
            execution,
            session_context.technical_requirements
        )

        if not is_valid:
            self.logger.warning(
                f"Execution {execution.id} validation failed with {len(validation_errors)} errors"
            )
            # Add validation errors to execution errors
            execution.errors.extend(validation_errors)
            # Mark as failed if major violations
            if any("❌" in error for error in validation_errors):
                execution.status = ExperimentStatus.FAILED
                self.logger.error(f"Execution {execution.id} marked as FAILED due to requirement violations")

return executions
```

### Step 7.3: Pass technical_requirements through OpenHandsSessionContext

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: `OpenHandsSessionContext` dataclass (around line 238)

Add field:
```python
@dataclass
class OpenHandsSessionContext:
    session_id: str
    workspace_id: str
    container_name: str
    research_id: str
    technical_requirements: Optional[TechnicalRequirements] = None  # NEW
```

**Update where context is created** (around line 2986):

```python
openhands_context = OpenHandsSessionContext(
    session_id=openhands_session_id,
    workspace_id=workspace_id,
    container_name=container_name,
    research_id=research_id,
    technical_requirements=result.technical_requirements  # NEW
)
```

## Phase 8: Update Smart Router Integration

### Step 8.1: Modify smart router to extract requirements

**File**: `backend/app/routers/smart_router.py`
**Location**: In `execute_scientific()` function (around line 303)

Add BEFORE calling `conduct_research`:

```python
async def execute_scientific() -> Dict[str, Any]:
    engine = engines["scientific"]
    params = classification_result.workflow_plan
    streaming_llm_client = StreamingLLMClient(base_llm_client, session_id)
    original_llm_client = engine.llm_client
    engine.llm_client = streaming_llm_client

    # Enable OpenHands streaming if available
    if hasattr(engine, 'openhands_client') and engine.openhands_client:
        engine.openhands_client = enable_openhands_streaming(engine.openhands_client, session_id)

    # NEW: Extract technical requirements from user request
    technical_requirements = None
    if hasattr(engine, 'requirement_extractor'):
        try:
            technical_requirements = await engine.requirement_extractor.extract_requirements(
                request.user_request
            )
            logger.info(f"Extracted technical requirements for session {session_id}: {technical_requirements.to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to extract technical requirements: {e}")

    try:
        execution_result = await engine.conduct_research(
            request.user_request,
            include_literature_review=params.get("include_literature_review", True),
            include_code_analysis=params.get("include_code_analysis", True),
            enable_iteration=params.get("enable_iteration", True),
            session_id=session_id,
            technical_requirements=technical_requirements  # NEW
        )
    finally:
        engine.llm_client = original_llm_client
```

## Phase 9: Add README.md Generation Requirement

### Step 9.1: Update FINAL DELIVERABLE section in comprehensive prompt

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `_build_comprehensive_experiment_prompt` method (around line 1244-1290)

**Current final deliverable section**:
```python
FINAL DELIVERABLE (REQUIRED)
═══════════════════════════════════════════════════════════════
Save a comprehensive final.json in /workspace/experiments/{plan.id}/results/final.json with:
...
IMPORTANT: The final.json MUST contain REAL data from actual execution, not synthetic/simulated data.
"""
```

**Change to add README.md requirement**:

```python
FINAL DELIVERABLE (REQUIRED)
═══════════════════════════════════════════════════════════════

1. Save a comprehensive final.json in /workspace/experiments/{plan.id}/results/final.json with:

{{
  "success": true/false,
  "steps_completed": [list of step numbers that completed successfully, e.g., [1, 2, 3]],
  "total_steps": {len(plan.experiments)},
  "data": {{
    "step_1": {{
      "measurements": [...],
      "artifacts": [...],
      "output_files": [...]
    }},
    ...
  }},
  "analysis": {{
    "approach": "High-level description of what approach was taken",
    "source_code_modifications": ["List paths to actual source files that were modified"],
    "build_artifacts": ["List of compiled binaries or build outputs"],
    "execution_logs": "Path to execution logs showing actual runs",
    "modifications_made": ["Specific changes made to achieve the objective"],
    "limitations": ["Any limitations encountered during execution"]
  }},
  "conclusions": [
    "Key finding 1 from the complete experimental sequence",
    "Key finding 2",
    ...
  ],
  "measurements": [
    actual_measured_values_as_numbers
  ],
  "reproducibility": {{
    "can_reproduce": true/false,
    "source_repositories": ["URLs of source code used"],
    "build_commands": ["Commands used to build software"],
    "reproduction_steps": ["Detailed steps to reproduce the entire experiment"]
  }}
}}

IMPORTANT: The final.json MUST contain REAL data from actual execution, not synthetic/simulated data.


2. Create a detailed README.md in /workspace/experiments/{plan.id}/README.md with:

# Experiment Reproduction Guide

## Overview
[Brief description of what this experiment does and what it proves]

## Experiment ID
- **Experiment ID**: {plan.id}
- **Date**: [Execution date]
- **Status**: [Success/Failed]
- **Hypothesis Tested**: [Hypothesis statement]

## Prerequisites

### System Requirements
- Operating System: [e.g., Ubuntu 20.04, macOS 13+]
- RAM: [e.g., 16GB minimum]
- Disk Space: [e.g., 50GB free space]
- CPU: [e.g., 4 cores minimum]

### Software Dependencies
List ALL software that must be installed, with specific versions:
```bash
# Example:
- PostgreSQL 16+ (built from source)
- Python 3.9+
- GCC 11+
- Make 4.3+
- Git 2.30+
```

### External Services
List any required external services:
- Network access to: [e.g., github.com, pypi.org]
- Proxy configuration: [if applicable]
- API keys needed: [if any]

## Directory Structure

Explain the directory layout created by the experiment:
```
/workspace/experiments/{plan.id}/
├── shared/                    # Shared installations and dependencies
│   ├── postgresql_install/   # PostgreSQL built from source
│   ├── duckdb_install/       # DuckDB installation
│   └── ...
├── step_1/                   # Step 1 outputs
│   ├── data/
│   ├── logs/
│   └── results/
├── step_2/                   # Step 2 outputs
├── results/                  # Final combined results
│   ├── final.json
│   ├── measurements.csv
│   └── analysis_report.txt
└── README.md                 # This file
```

## Step-by-Step Reproduction Instructions

### Step 0: Environment Setup

Provide exact commands to set up the environment:
```bash
# Clone repositories
git clone https://github.com/postgres/postgres.git /workspace/postgres
git clone https://github.com/duckdb/duckdb.git /workspace/duckdb

# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential libreadline-dev zlib1g-dev

# Set environment variables
export POSTGRES_HOME=/workspace/experiments/{plan.id}/shared/postgresql_install
export PATH=$POSTGRES_HOME/bin:$PATH
```

### Step 1: [First Experiment Step Name]

**Goal**: [What this step accomplishes]

**Commands**:
```bash
# Provide EXACT commands executed in this step
cd /workspace/postgres
./configure --prefix=/workspace/experiments/{plan.id}/shared/postgresql_install
make -j$(nproc)
make install
```

**Expected Output**:
```
[Show expected output/logs from successful execution]
```

**Generated Files**:
- `/workspace/experiments/{plan.id}/shared/postgresql_install/bin/postgres` - PostgreSQL binary
- `/workspace/experiments/{plan.id}/step_1/build.log` - Build log

**Verification**:
```bash
# Commands to verify step completed successfully
/workspace/experiments/{plan.id}/shared/postgresql_install/bin/postgres --version
# Expected: postgres (PostgreSQL) 16.x
```

### Step 2: [Second Experiment Step Name]

**Goal**: [What this step accomplishes]

**Commands**:
```bash
# Provide EXACT commands
[...]
```

**Expected Output**:
```
[...]
```

**Generated Files**:
- [List all files created]

**Verification**:
```bash
# Verification commands
[...]
```

[Repeat for each step...]

## Modified Source Code Files

List ALL source code files that were modified, with brief description:

1. **File**: `/workspace/postgres/src/backend/optimizer/plan/planner.c`
   - **Lines Modified**: 280-320
   - **Purpose**: Added ML routing hook to planner entry point
   - **Changes**:
     ```c
     // Added before standard_planner() call
     if (ml_routing_enabled) {
         MLRoutingDecision decision = ml_route_query(parse, query_string);
         if (decision.route_to_duckdb) {
             result = duckdb_execute_query(parse, query_string);
             return result;
         }
     }
     ```

2. **File**: `/workspace/postgres/contrib/ml_router/ml_router.c`
   - **Lines Modified**: 1-200 (new file)
   - **Purpose**: C implementation of embedded ML model for query routing
   - **Changes**: [Full file created with prediction function]

[List all other modified files...]

## Data Files Generated

### Training Data
- **Location**: `/workspace/experiments/{plan.id}/step_2/training_data.csv`
- **Size**: [e.g., 1.2 MB, 5000 rows]
- **Format**: CSV with columns [list columns]
- **Sample**:
  ```csv
  query_id,feature1,feature2,...,pg_time_ms,duckdb_time_ms,faster_engine
  q001,0.5,0.3,...,45.2,12.1,duckdb
  ```

### Model Files
- **Location**: `/workspace/experiments/{plan.id}/step_3/model.pkl`
- **Type**: scikit-learn RandomForestClassifier
- **Size**: [e.g., 2.4 MB]
- **Accuracy**: [e.g., 92.5% on test set]

### Results
- **Location**: `/workspace/experiments/{plan.id}/results/final.json`
- **Contains**: Complete experiment results and measurements

## Running the Integrated System

After completing all steps, here's how to run the integrated ML-based routing system:

```bash
# Start PostgreSQL with ML routing enabled
cd /workspace/experiments/{plan.id}/shared/postgresql_install
./bin/pg_ctl -D data -l logfile start
./bin/psql -d testdb -c "SET ml_routing_enabled = on;"

# Run test queries
./bin/psql -d testdb -f /workspace/experiments/{plan.id}/test_queries.sql

# Check routing decisions in logs
tail -f /workspace/experiments/{plan.id}/shared/postgresql_install/data/logfile | grep "ML_ROUTING"
```

**Expected Behavior**:
- Simple queries (low complexity) route to PostgreSQL
- Complex analytical queries (high complexity) route to DuckDB
- Routing decision logged for each query

## Verification and Testing

### Verify Source Code Modifications
```bash
# Check that modifications are present
grep -n "ml_routing_enabled" /workspace/postgres/src/backend/optimizer/plan/planner.c
# Should show modified lines

# Verify ML extension is installed
/workspace/experiments/{plan.id}/shared/postgresql_install/bin/psql -d postgres -c "SELECT * FROM pg_extension WHERE extname='ml_router';"
```

### Verify Data Collection
```bash
# Check dual-execution data was collected
wc -l /workspace/experiments/{plan.id}/step_2/training_data.csv
# Should show >100 rows

# Verify both engines were used
cut -d',' -f7 /workspace/experiments/{plan.id}/step_2/training_data.csv | sort | uniq
# Should show both "postgres" and "duckdb"
```

### Verify Model Embedding
```bash
# Check ML model was embedded in C code
ls -lh /workspace/experiments/{plan.id}/shared/postgresql_install/lib/ml_router.so
# Should exist and be >100KB

# Test prediction function
/workspace/experiments/{plan.id}/shared/postgresql_install/bin/psql -d postgres -c "SELECT ml_predict_engine(1.0, 2.0, 3.0);"
# Should return 0 or 1 (engine prediction)
```

## Results Summary

### Key Findings
[Summarize main experimental findings from final.json conclusions]

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Performance Metrics
[Include key measurements from final.json]

- **Routing Decision Time**: [e.g., 0.5ms average]
- **Prediction Accuracy**: [e.g., 92.5%]
- **Throughput Impact**: [e.g., <2% overhead]

### Comparison: Before vs After
| Metric | Before ML Routing | After ML Routing | Improvement |
|--------|------------------|------------------|-------------|
| Avg Query Time | [X ms] | [Y ms] | [Z%] |
| [Other metrics] | | | |

## Troubleshooting

### Common Issues

**Issue 1: PostgreSQL fails to build**
```
Error: configure: error: readline library not found
```
**Solution**:
```bash
sudo apt-get install -y libreadline-dev
```

**Issue 2: Permission denied when running PostgreSQL**
```
Error: could not create lock file: Permission denied
```
**Solution**:
```bash
# Use user-space installation
initdb -D $HOME/pgdata
```

**Issue 3: DuckDB connection fails**
```
Error: could not connect to DuckDB
```
**Solution**:
```bash
# Check DuckDB is installed
pip install duckdb
```

[Add more common issues encountered during your execution]

## Differences from Original Plan

[If any steps deviated from the original plan, explain why]

- **Deviation 1**: [What changed and why]
- **Deviation 2**: [What changed and why]

## Limitations

[List any limitations from final.json analysis.limitations]

1. [Limitation 1]
2. [Limitation 2]

## Future Improvements

[Suggest potential improvements to the experiment]

1. [Improvement 1]
2. [Improvement 2]

## Contact and References

### Source Repositories
[List all repositories used, from final.json reproducibility.source_repositories]
- PostgreSQL: https://github.com/postgres/postgres
- DuckDB: https://github.com/duckdb/duckdb
- [Others...]

### Build Commands Reference
[Complete list from final.json reproducibility.build_commands]

### Experiment Metadata
- **Hypothesis ID**: [ID]
- **Execution Date**: [Date]
- **Total Duration**: [Time]
- **OpenHands Session**: [Session ID if available]

---

**Note**: This README was automatically generated based on the actual experiment execution. All commands and outputs are from the real execution, not simulated.


CRITICAL REQUIREMENTS FOR README.md:

1. ✅ README.md MUST be created in /workspace/experiments/{plan.id}/README.md
2. ✅ Must contain EXACT commands that were actually executed (not generic examples)
3. ✅ Must list ALL modified source code files with specific line numbers
4. ✅ Must show REAL output/logs from execution (not placeholder text)
5. ✅ Must enable anyone to reproduce the experiment from scratch
6. ✅ Must include verification commands to check each step succeeded
7. ✅ Must document all deviations from the original plan
8. ✅ Must include troubleshooting section for issues encountered
9. ✅ README.md is MANDATORY - experiments without it are considered incomplete

"""
```

### Step 9.2: Update validation to check for README.md

**File**: `backend/app/core/research_engines/scientific_research.py`
**Location**: In `RequirementValidator.validate_execution_results` method

Add check for README.md after checking for output_data:

```python
async def validate_execution_results(
    self,
    execution: ExperimentExecution,
    requirements: TechnicalRequirements
) -> Tuple[bool, List[str]]:
    """Validate execution results meet technical requirements after execution"""

    validation_errors = []

    # Check output_data for compliance
    final_data = execution.output_data
    if not final_data:
        validation_errors.append("❌ No output data found in execution results")
        return False, validation_errors

    # NEW: Check for README.md file
    files_generated = final_data.get("data", {}).get("files_generated", [])
    readme_files = [f for f in files_generated if "README.md" in f or "readme.md" in f]

    if not readme_files:
        validation_errors.append(
            "❌ MISSING REQUIRED README.md - Experiment must generate detailed reproduction guide"
        )

    # ... rest of validation ...
```

## Testing Plan

### Test 1: Requirement Extraction

Create test: `backend/test/unit/test_requirement_extractor.py`

```python
import pytest
from app.core.research_engines.scientific_research import RequirementExtractor, TechnicalRequirements

@pytest.mark.asyncio
async def test_extract_requirements_postgres_duckdb(llm_client_mock):
    extractor = RequirementExtractor(llm_client_mock)

    research_question = """
    please modify postgres and pg_duckdb source code, first extract pre-opt features
    from postgres kernel and log to files, then collect dual-execution data (pre-optimization
    query features that can be found in kernel structures and execution times on dual engine)
    and train a machine learning model to predict whether postgres engine or duckdb engine
    executes a query fast and embed the machine learning model into database source code
    (using the language of the database for example c language) to online route each query
    to the faster engine
    """

    requirements = await extractor.extract_requirements(research_question)

    assert "PostgreSQL" in str(requirements.source_code_modifications)
    assert "pg_duckdb" in str(requirements.source_code_modifications)
    assert "C" in requirements.programming_languages or "c language" in str(requirements.programming_languages)
    assert len(requirements.execution_engines) >= 2
    assert "PostgreSQL" in requirements.execution_engines or "postgres" in str(requirements.execution_engines)
    assert "DuckDB" in requirements.execution_engines or "duckdb" in str(requirements.execution_engines)
    assert any("embed" in req.lower() for req in requirements.integration_requirements)
    assert any("dual" in req.lower() for req in requirements.data_collection_requirements)
```

### Test 2: Experiment Plan Validation

Create test: `backend/test/unit/test_requirement_validator.py`

```python
import pytest
from app.core.research_engines.scientific_research import (
    RequirementValidator,
    TechnicalRequirements,
    SequentialExperimentPlan,
    ExperimentDesign
)

@pytest.mark.asyncio
async def test_reject_rest_api_when_prohibited():
    validator = RequirementValidator()

    requirements = TechnicalRequirements(
        prohibited_shortcuts=["no REST APIs", "no external APIs"]
    )

    plan = SequentialExperimentPlan(
        id="test",
        hypothesis_id="hyp1",
        num_experiments=1,
        experiments=[
            ExperimentDesign(
                id="exp1",
                hypothesis_id="hyp1",
                name="Test Experiment",
                description="Deploy model via Flask REST API",
                methodology="Step 1: Create REST API endpoint",
                variables={},
                controls=[],
                data_collection_plan={},
                analysis_plan="",
                expected_duration="1h",
                resource_requirements={},
                code_requirements=[],
                dependencies=[]
            )
        ],
        overall_objective="Test",
        experiment_dependencies={},
        shared_setup="",
        expected_total_duration="1h"
    )

    is_valid, errors = await validator.validate_experiment_plan(plan, requirements)

    assert not is_valid
    assert len(errors) > 0
    assert any("REST API" in error or "external API" in error for error in errors)
```

### Test 3: End-to-End with Real User Query

Create test: `backend/test/e2e/test_requirement_compliance.py`

```python
import pytest
from app.core.research_engines.scientific_research import ScientificResearchEngine

@pytest.mark.asyncio
@pytest.mark.slow
async def test_postgres_duckdb_experiment_complies_with_requirements(
    llm_client,
    openhands_client
):
    engine = ScientificResearchEngine(
        llm_client=llm_client,
        openhands_client=openhands_client
    )

    research_question = """
    modify postgres and pg_duckdb source code, extract pre-optimization features
    from postgres kernel, collect dual-execution data, train ML model, and embed
    the model into database source code using C language
    """

    result = await engine.conduct_research(
        research_question,
        include_literature_review=False,
        include_code_analysis=False,
        enable_iteration=False
    )

    # Verify requirements were extracted
    assert result.technical_requirements is not None
    assert len(result.technical_requirements.execution_engines) >= 2

    # Verify experiments were generated
    assert len(result.experiments) > 0

    # Verify experiments don't use prohibited shortcuts
    for execution in result.executions:
        final_data_str = str(execution.output_data).lower()
        assert "rest api" not in final_data_str or "flask" not in final_data_str
        assert "synthetic" not in final_data_str or "simulated" not in final_data_str

        # Verify dual-engine execution
        assert ("postgres" in final_data_str and "duckdb" in final_data_str)
```

## Rollout Strategy

### Phase 1: Development (Days 1-3)
1. Implement RequirementExtractor class
2. Implement RequirementValidator class
3. Implement TechnicalGuideProvider class
4. Write unit tests

### Phase 2: Integration (Days 4-6)
5. Integrate into HypothesisGenerator
6. Integrate into ExperimentDesigner
7. Add validation loops
8. Write integration tests

### Phase 3: Testing (Days 7-8)
9. Run end-to-end tests with user's actual query
10. Verify experiments comply with requirements
11. Fix any remaining issues

### Phase 4: Deployment (Day 9)
12. Deploy to production
13. Monitor experiment success rates
14. Collect feedback

## Success Criteria

After implementation, experiments should:

1. ✅ Extract technical requirements from user queries with >90% accuracy
2. ✅ Generate hypotheses that explicitly test dual-engine execution
3. ✅ Design experiments that require source code modification
4. ✅ Reject experiment designs that violate prohibited shortcuts
5. ✅ Provide technical guidance for complex implementation tasks
6. ✅ Validate experiments before and after execution
7. ✅ Achieve >80% compliance rate with user's technical requirements
8. ✅ Reduce synthetic data usage to <10% of experiments
9. ✅ Reduce REST API usage (when prohibited) to 0%
10. ✅ Increase source code modification rate to >90% when required

## Monitoring and Metrics

Track these metrics after deployment:

1. **Requirement Extraction Accuracy**: % of correctly extracted requirements
2. **Validation Rejection Rate**: % of experiment plans rejected in pre-execution validation
3. **Requirement Compliance Rate**: % of executions that pass post-execution validation
4. **Synthetic Data Usage**: % of experiments using synthetic data
5. **Source Code Modification Rate**: % of experiments that modify source code when required
6. **Dual-Engine Execution Rate**: % of experiments that use both engines when required
7. **REST API Usage**: % of experiments using REST APIs when prohibited
8. **User Satisfaction**: User feedback on experiment quality

## Conclusion

This implementation plan provides a systematic approach to fixing the experiment compliance issues. The key innovations are:

1. **Requirement extraction** at the start of research
2. **Constraint injection** into all LLM prompts
3. **Validation loops** before and after execution
4. **Technical guidance** for complex implementation tasks

Together, these changes ensure that experiments follow the user's exact technical requirements rather than taking generic shortcuts.
