# OCC Validator - Optimistic Concurrency Control for Multi-Agent Code Editing

## Overview

The OCC (Optimistic Concurrency Control) Validator implements database-style optimistic concurrency control for multi-agent parallel code editing in OpenHands. It tracks read-set and write-set during agent execution and validates conflicts at commit time using AST region signatures.

## Architecture

### Core Components

1. **OCCValidator** (`openhands/core/occ_validator.py`)
   - Main validation logic
   - AST region signature computation
   - Conflict detection and resolution

2. **OCCValidatorManager** (`openhands/core/occ_validator_integration.py`)
   - Multi-agent coordination
   - Per-agent tracking
   - Lifecycle management

3. **OCC Events** (`openhands/events/observation/occ.py`)
   - Observability for OCC operations
   - Tracking, validation, and conflict events

## Key Features

### ✅ Implemented

- **AST-Based Conflict Detection**: Uses Abstract Syntax Tree analysis to detect semantic conflicts beyond textual differences
- **Region-Level Tracking**: Tracks specific code regions (functions, classes, methods) rather than entire files
- **Language Support**: Python (full), JavaScript/TypeScript (partial), Java (partial)
- **Signature Caching**: Caches AST signatures for performance
- **Conflict Classification**: Distinguishes between formatting, AST reordering, semantic, read-write, and write-write conflicts
- **Event Emission**: Rich observability through OCC observation events
- **Tree-sitter Integration**: Enhanced parsing with tree-sitter (with fallback to Python's ast module)
- **Robust String/Comment Handling**: Ignores braces in strings and comments during block detection

### 🎯 Recent Improvements (v1.1)

1. **Fixed AST Signature File Path**: Now passes actual file paths to enable proper language detection
2. **Agent ID Tracking**: All OCC events now include agent_id for proper observability
3. **Enhanced Region Matching**: Matches regions by name, type, and line-range overlap to disambiguate same-named entities
4. **Tree-sitter Fallback Warnings**: One-time warnings when AST parsers unavailable
5. **Improved Block Detection**: Enhanced regex-based block detection to handle braces in strings/comments
6. **Clean API Signatures**: Integration hooks use `functools.wraps` to preserve original method signatures
7. **Consolidated Event Emission**: Events emitted from validator with agent_id context

## Integration Guide

### Basic Setup

```python
from openhands.core.occ_validator_integration import (
    OCCValidatorManager,
    install_occ_hooks,
    OCCConfig
)
from openhands.core.dependency_analyzer import DependencyAnalyzer
from openhands.runtime.utils.git_handler import GitHandler

# Create dependencies
git_handler = GitHandler(workspace_base)
dependency_analyzer = DependencyAnalyzer()

# Create OCC manager
occ_manager = OCCValidatorManager(
    workspace_base=workspace_base,
    workspace_mount_path_in_sandbox=workspace_mount_path,
    dependency_analyzer=dependency_analyzer,
    git_handler=git_handler,
    logger=logger,
    event_stream=event_stream,
    enable_tracking=True
)

# Install hooks in coordinator (preserves original API)
install_occ_hooks(coordinator, occ_manager)
```

### Configuration

```python
from openhands.core.occ_validator_integration import OCCConfig

# Default configuration (strict validation)
config = OCCConfig.default()

# Permissive configuration (auto-rebase on minor conflicts)
config = OCCConfig.permissive()

# Disabled configuration
config = OCCConfig.disabled()

# Custom configuration
config = OCCConfig(
    enable_occ_validation=True,
    occ_validation_mode='strict',  # or 'permissive'
    occ_auto_rebase=False,
    occ_track_read_set=True,
    occ_track_write_set=True,
    occ_max_validation_time_ms=10000.0,
    occ_conflict_resolution_strategy='conservative'  # or 'aggressive'
)
```

### Manual Validation

```python
# Start tracking for an agent
agent_id = "agent_001"
base_commit = git_handler.get_current_commit()
occ_manager.start_tracking(agent_id, base_commit)

# Get tracker for manual read/write tracking
tracker = occ_manager.get_agent_tracker(agent_id)
tracker.track_file_read("src/main.py")
tracker.track_file_write("src/utils.py")

# Validate agent's changes
current_head = git_handler.get_current_commit()
result = await occ_manager.validate_agent_commit(agent_id, current_head)

if result.success:
    print("✓ No conflicts detected")
else:
    print(f"✗ {result.conflicts_count} conflicts detected")
    for conflict in result.conflicts:
        print(f"  - {conflict.file_path}: {conflict.conflict_type.value}")
    print(f"Suggested resolution: {result.suggested_resolution}")

# Cleanup
occ_manager.cleanup_agent(agent_id)
```

## Conflict Types

### 1. FORMATTING
Whitespace or formatting differences only (semantic equivalence)
- **Resolution**: Usually safe to auto-rebase

### 2. AST_REORDERING
Reordering of imports, class members, etc. without semantic changes
- **Resolution**: Can often auto-rebase

### 3. SEMANTIC
Changes to logic, control flow, or data structures
- **Resolution**: Requires manual review

### 4. READ_WRITE
Agent read a region that was subsequently modified by another agent
- **Resolution**: May require rebase or merge

### 5. WRITE_WRITE
Multiple agents modified the same region
- **Resolution**: Requires manual conflict resolution or abort

## OCC Events

### Tracking Events
- `OCCTrackingStartedObservation`: Emitted when tracking starts for an agent
- `OCCFileAccessObservation`: Emitted on file read/write operations

### Validation Events
- `OCCValidationStartedObservation`: Validation begins
- `OCCValidationCompletedObservation`: Validation completes with results
- `OCCConflictDetectedObservation`: Individual conflict detected

### Rebase Events
- `OCCAutoRebaseStartedObservation`: Auto-rebase initiated
- `OCCAutoRebaseCompletedObservation`: Auto-rebase completed

### Analysis Events
- `OCCRegionAnalysisObservation`: Detailed region analysis performed
- `OCCCacheHitObservation`: Cache hit during validation

## API Reference

### OCCValidator

```python
class OCCValidator:
    async def validate_commit(
        self,
        read_set: Dict[str, Set[Region]],
        write_set: Dict[str, Set[Region]],
        base_commit: str,
        current_head: str,
        agent_id: str = ""
    ) -> ValidationResult:
        """Validate agent's changes against current state."""
```

### OCCValidatorManager

```python
class OCCValidatorManager:
    def start_tracking(self, agent_id: str, base_commit: str):
        """Start tracking for an agent."""
    
    async def validate_agent_commit(
        self, 
        agent_id: str, 
        current_head: str
    ) -> ValidationResult:
        """Validate agent's changes."""
    
    def cleanup_agent(self, agent_id: str):
        """Remove agent's validator and tracker."""
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for specific agent."""
```

### Region

```python
@dataclass
class Region:
    file_path: str
    start_line: int
    end_line: int
    region_type: Literal['function', 'class', 'method', 'block', 'statement']
    name: str
    signature: str = ""
    
    def overlaps(self, other: 'Region') -> bool:
        """Check if this region overlaps with another region."""
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    success: bool
    conflicts: List[ConflictDetail] = field(default_factory=list)
    suggested_resolution: Literal['commit', 'rebase', 'merge', 'abort'] = 'commit'
    validation_duration_ms: float = 0.0
    read_conflicts_count: int = 0
    write_conflicts_count: int = 0
```

## Performance Considerations

### Caching
- AST signatures are cached per file region
- Cache key: `file_path:start_line:end_line:content_hash`
- Clear cache between major refactorings

### Validation Time
- Average validation: < 100ms for typical changes
- Large changesets: 500ms - 2000ms
- Configure `occ_max_validation_time_ms` to timeout long validations

### Resource Usage
- Memory: ~5-10MB per tracked agent
- CPU: Minimal overhead during tracking, spike during validation

## Limitations

1. **Language Support**: Full support for Python only; JavaScript/Java use basic parsing
2. **Tree-sitter Dependency**: Enhanced features require tree-sitter installation
3. **Cross-file Dependencies**: Currently tracks only direct file changes (transitive analysis reserved for future)
4. **Large Files**: Files > 10,000 lines may have slower validation
5. **Binary Files**: Only text files supported; binary files treated as whole-file regions

## Troubleshooting

### Tree-sitter Not Available
```
WARNING: Tree-sitter not available. Install tree-sitter and tree-sitter-languages for enhanced AST parsing.
```
**Solution**: Install tree-sitter support
```bash
pip install tree-sitter tree-sitter-languages
```

### AST Parsing Failures
**Symptom**: Validation falls back to normalized content hashing
**Cause**: Syntax errors in code or unsupported language constructs
**Solution**: Ensure code is syntactically valid; check language support

### Empty agent_id in Events
**Symptom**: OCC events show empty agent_id
**Solution**: Ensure `validate_commit` is called with agent_id parameter (fixed in v1.1)

### False Conflicts
**Symptom**: Conflicts reported for non-conflicting changes
**Cause**: May be formatting or region matching issue
**Solution**: Use permissive mode or adjust conflict resolution strategy

## Best Practices

1. **Track Granularly**: Use region-level tracking when possible for better conflict detection
2. **Validate Early**: Validate before committing to catch conflicts early
3. **Handle Conflicts Gracefully**: Implement proper conflict resolution UI/UX
4. **Monitor Events**: Subscribe to OCC events for observability
5. **Configure Appropriately**: Use strict mode for production, permissive for development
6. **Clean Up**: Always call `cleanup_agent` to free resources

## Future Enhancements

- **Transitive Dependency Analysis**: Leverage DependencyAnalyzer for cross-file impact
- **Smart Merge Strategies**: Auto-merge compatible changes
- **Conflict Visualization**: UI for conflict resolution
- **Multi-language Support**: Full support for JavaScript, TypeScript, Java, Go, Rust
- **Distributed Validation**: Support for distributed multi-agent scenarios

## Version History

### v1.1 (Current)
- Fixed AST signature file path parameter
- Added agent_id to all OCC events
- Enhanced region matching with type checks
- Tree-sitter fallback warnings
- Improved block detection (string/comment handling)
- Clean API with functools.wraps
- Consolidated event emission

### v1.0
- Initial implementation
- Basic conflict detection
- AST region signatures
- Multi-agent tracking

## Contributing

See the main OpenHands contributing guide. For OCC-specific changes:
1. Update tests in `tests/unit/test_occ_validator.py`
2. Update integration tests in `tests/integration/test_occ_integration.py`
3. Update this README
4. Ensure backward compatibility

## License

Same as OpenHands project license.
