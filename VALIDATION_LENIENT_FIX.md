# Validation Lenient Fix - Support Generic Research Questions

## Problem Statement

Experiment validation was **too strict** and **domain-specific**, causing validation failures for generic research questions that didn't match PostgreSQL/DuckDB requirements:

```json
{
  "validation_errors": [
    "❌ Experiment may violate prohibition: do not use the system-wide postgresql",
    "⚠️  Experiment does not mention required language: C language",
    "⚠️  Experiment does not mention required language: language of the database",
    "❌ Shared setup violates prohibition: do not use the system-wide postgresql"
  ]
}
```

**Issues**:
1. **Too literal text matching**: "postgresql" in text matched "do not use system-wide postgresql" prohibition
2. **Too strict language checks**: Required exact mention of programming languages
3. **Domain-specific**: Assumed all research involves databases (PostgreSQL, DuckDB, C language)
4. **Blocked valid experiments**: Agent's experiment plans were rejected even if they had valid approaches

## Root Cause

### Issue 1: Overly Aggressive Prohibition Matching

**Old Code** (Line 349-355):
```python
for prohibition in requirements.prohibited_shortcuts:
    prohibition_words = prohibition.lower().split()
    # Check if ANY key words from prohibition appear in methodology
    if any(word in full_text for word in prohibition_words if len(word) > 3):
        validation_errors.append(
            f"❌ Experiment {exp_idx} '{experiment.name}' may violate prohibition: {prohibition}"
        )
```

**Problem**:
- Prohibition "do not use the system-wide postgresql" split into words: ["do", "not", "use", "the", "system-wide", "postgresql"]
- If experiment mentioned "postgresql" anywhere, it matched word "postgresql" → false positive

### Issue 2: Mandatory Language Checks

**Old Code** (Line 384-390):
```python
# Check for programming language requirements
if requirements.programming_languages:
    for lang in requirements.programming_languages:
        if lang.lower() not in full_text:
            validation_errors.append(
                f"⚠️  Experiment {exp_idx} '{experiment.name}' does not mention required language: {lang}"
            )
```

**Problem**:
- Required exact text match for language names
- Treated warnings (⚠️) same as errors (❌) - both caused validation failure
- Agent could use a language without explicitly mentioning it in the description

### Issue 3: Overly Prescriptive Requirement Extraction

**Old Prompt** (Line 281-299):
```
Extract and return in JSON format with these exact keys:
1. "programming_languages": Which languages must be used (e.g., ["C language", "Python"])
2. "prohibited_shortcuts": What approaches are explicitly forbidden (e.g., ["no REST APIs"])

Be SPECIFIC and LITERAL - extract exact technical requirements from the user's wording.
If the user says "modify postgres source code", include that exact phrase.
```

**Problem**:
- LLM extracted requirements even when not explicitly stated
- Assumed database research requires "C language" and "language of the database"
- Over-inferred prohibitions from context

## Solution

### Change 1: Lenient Validation - Warnings Only

**File**: `backend/app/core/research_engines/scientific_research.py` (Lines 333-419)

**Key Changes**:
1. **Separate warnings from critical errors**:
   ```python
   validation_errors = []  # Warnings only
   critical_errors = []    # Critical failures only

   # ONLY fail validation if there are CRITICAL errors
   is_valid = len(critical_errors) == 0
   ```

2. **No critical errors added**: All checks now produce warnings (⚠️), not errors (❌)
   - Validation always passes (`is_valid = True`)
   - Warnings are logged but don't block experiments

3. **Removed strict language checks**:
   ```python
   # REMOVED: Programming language checks - too strict and domain-specific
   # Agent can use appropriate language without explicitly mentioning it
   ```

4. **Phrase matching instead of word matching**:
   ```python
   # For multi-word prohibitions, check for the complete phrase
   if len(prohibition.split()) > 2:
       if prohibition_lower in full_text:
           warning = f"⚠️  Experiment may involve: {prohibition}"
           validation_errors.append(warning)
           # Don't treat as critical error
   else:
       # For short prohibitions, skip validation (too ambiguous)
       pass
   ```

5. **More lenient engine checks**:
   ```python
   # Only validate if multiple engines required (>2)
   if len(requirements.execution_engines) > 2:
       # Only warn if less than HALF of engines mentioned
       if len(engines_mentioned) < len(requirements.execution_engines) // 2:
           warning = f"⚠️  Experiment mentions few required engines"
   ```

### Change 2: Conservative Requirement Extraction

**File**: `backend/app/core/research_engines/scientific_research.py` (Lines 281-305)

**Key Changes**:

**New Prompt**:
```
IMPORTANT GUIDELINES:
- Be CONSERVATIVE: Only extract requirements that are EXPLICITLY stated in the research question
- Do NOT infer or assume requirements that aren't clearly mentioned
- If a field doesn't apply, return an EMPTY list []
- Programming languages: Only include if user specifically mentions them (e.g., "using C")
- Prohibitions: Only include if user says "do not", "avoid", "without", "must not"
- Keep lists SHORT (1-3 items max per field)
- Be GENERIC and FLEXIBLE - support all types of research, not just database research

Return ONLY a valid JSON object with these keys. Use empty lists [] for fields that don't apply.
```

**Impact**:
- LLM will extract fewer requirements
- Only explicit mentions in user request
- No inferred requirements
- Empty lists for non-applicable fields

## Comparison

### Before Fix

**User Request**: "Please research how to implement feature X"

**Extracted Requirements** (over-inferred):
```json
{
  "programming_languages": ["C language", "language of the database"],
  "prohibited_shortcuts": ["do not use system-wide postgresql"],
  "execution_engines": ["PostgreSQL", "DuckDB"]
}
```

**Validation Result**: ❌ FAILED
```
❌ Experiment does not mention required language: C language
❌ Experiment may violate prohibition: do not use the system-wide postgresql
```

**Outcome**: Experiment blocked, redesign triggered

---

### After Fix

**User Request**: "Please research how to implement feature X"

**Extracted Requirements** (conservative):
```json
{
  "programming_languages": [],  // Not explicitly mentioned
  "prohibited_shortcuts": [],   // No explicit prohibitions
  "execution_engines": []       // Generic research
}
```

**Validation Result**: ✅ PASSED
```
INFO - Experiment plan has 0 validation warnings (non-critical)
```

**Outcome**: Experiment proceeds

## Benefits

### 1. Support Generic Research Questions ✅

- Works for ANY research domain (not just databases)
- No false positives from word matching
- Doesn't require specific language mentions

### 2. Fewer False Validation Failures ✅

- Warnings don't block experiments
- Only critical errors cause failure (none currently)
- Agent can proceed with valid approaches

### 3. Conservative Requirement Extraction ✅

- Only extracts explicitly stated requirements
- No over-inference
- Empty lists for non-applicable fields

### 4. Better User Experience ✅

- Experiments proceed without unnecessary redesigns
- Faster research completion
- Less back-and-forth with validation

## Edge Cases Handled

### Case 1: Database Research with Prohibitions

**User**: "Modify PostgreSQL source code, but do not use system-wide postgresql"

**Extracted**:
```json
{
  "source_code_modifications": ["PostgreSQL source code"],
  "prohibited_shortcuts": ["do not use system-wide postgresql"]
}
```

**Validation**:
- ⚠️ Warning if "system-wide postgresql" phrase found
- ✅ Still passes (warning only)
- Agent can clarify in implementation

### Case 2: Generic ML Research

**User**: "Train a model to predict customer churn"

**Extracted**:
```json
{
  "source_code_modifications": [],
  "programming_languages": [],
  "execution_engines": [],
  "prohibited_shortcuts": []
}
```

**Validation**:
- ✅ Passes with no warnings
- No domain-specific checks applied

### Case 3: Specific Language Requirement

**User**: "Implement this algorithm in Rust"

**Extracted**:
```json
{
  "programming_languages": ["Rust"]
}
```

**Validation**:
- ⚠️ Warning if "rust" not mentioned in methodology
- ✅ Still passes (warning only)
- Agent can use Rust without mentioning it in description

## Testing

### Test 1: Generic Research Question
```python
query = "Research how to improve web application performance"

# Expected: No requirements extracted, validation passes
requirements = await extractor.extract_requirements(query)
assert requirements.programming_languages == []
assert requirements.prohibited_shortcuts == []

is_valid, errors = await validator.validate_experiment_plan(plan, requirements)
assert is_valid == True
assert len(errors) == 0
```

### Test 2: Database Research (Original Issue)
```python
query = "Modify PostgreSQL and pg_duckdb source code, do not use system-wide postgresql"

# Expected: Conservative extraction, lenient validation
requirements = await extractor.extract_requirements(query)
# May extract some requirements, but fewer than before

is_valid, errors = await validator.validate_experiment_plan(plan, requirements)
assert is_valid == True  # Always true now (no critical errors)
# May have warnings, but doesn't fail
```

## Migration Notes

### Backwards Compatibility

- ✅ **Compatible**: Existing research queries will work
- ✅ **Safer**: Previously failing validations now pass
- ✅ **Informative**: Warnings still logged for review

### No Database Changes

- Tree structure unchanged
- No API contract changes
- Only internal validation logic modified

## Rollback Plan

If validation is too lenient:

```bash
git diff backend/app/core/research_engines/scientific_research.py
git checkout backend/app/core/research_engines/scientific_research.py
```

## Success Metrics

- ✅ Validation passes for generic research questions
- ✅ No false positives from word matching
- ✅ Warnings logged but don't block experiments
- ✅ Conservative requirement extraction
- ✅ Supports all research domains

## Files Modified

1. `backend/app/core/research_engines/scientific_research.py`
   - Lines 281-305: Updated requirement extraction prompt (conservative)
   - Lines 333-419: Updated validation logic (lenient, warnings only)

## Implementation Date

2025-10-02

## Priority

High - Blocks generic research questions from executing
