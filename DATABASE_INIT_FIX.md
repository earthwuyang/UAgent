# Database Initialization Fix

## Problem
The UAgent Research Extension was failing with the error:
```
RuntimeError: Database not initialized. Call init_database() first.
```

This occurred when API endpoints tried to access the database through the `get_session()` dependency.

## Root Cause
The issue was caused by an **inconsistent Python path setup** between different modules:

1. **`app.py`** (line 44): Added `extensions/uagent_research` to sys.path
2. **`listen.py`** (line 11): Added only `extensions` to sys.path

This inconsistency meant that when modules imported `uagent_research`, they could be getting different module instances, causing the global `_async_session_factory` variable (which is set during database initialization) to be `None` in some contexts.

## The Fix

### Changed Files

#### 1. `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/listen.py`
**Change**: Updated the path to match `app.py`'s import style

```python
# Before:
extensions_dir = Path(__file__).parent.parent.parent / 'extensions'

# After:
extension_dir = Path(__file__).parent.parent.parent / 'extensions' / 'uagent_research'
```

This ensures that both `app.py` and `listen.py` add the same path to `sys.path`, so all imports of `uagent_research` resolve to the same module instance.

#### 2. `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/app.py`
**Change**: Enhanced logging to help diagnose initialization issues

```python
# Added better logging around database initialization
logger.info(f"🔄 Initializing research database: {research_db_url}")
await init_database(research_db_url, echo=False)
logger.info(f"✅ Research database initialized successfully")
```

## How Database Initialization Works

The database is initialized through FastAPI's **lifespan context manager** in `app.py`:

```python
@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize UAgent Research Extension database if available
    if RESEARCH_EXTENSION_AVAILABLE:
        research_db_url = os.getenv(
            'RESEARCH_DATABASE_URL',
            'sqlite+aiosqlite:///./openhands_research.db'
        )
        await init_database(research_db_url, echo=False)
    
    async with conversation_manager:
        yield
    
    # Cleanup on shutdown
    if RESEARCH_EXTENSION_AVAILABLE:
        await close_database()
```

This lifespan runs **before** the application starts accepting requests, ensuring the database is ready when the first request comes in.

## Verification

After the fix, the server starts successfully with logs showing:
```
🔄 Initializing research database: sqlite+aiosqlite:///./openhands_research.db
✅ Research database initialized successfully
```

And API endpoints work correctly:
```bash
$ curl --noproxy "*" http://localhost:2999/api/research/health
{"status":"healthy","extension":"uagent_research","version":"0.1.0","timestamp":"2025-10-12T07:04:34.362446"}

$ curl --noproxy "*" http://localhost:2999/api/research/experiments
[...list of experiments from database...]
```

## Key Takeaways

1. **Module Import Consistency**: When using `sys.path` manipulation, ensure all modules use the same paths to avoid module instance duplication
2. **Global State**: Be careful with global state (like `_async_session_factory`) - it only works if all imports resolve to the same module
3. **FastAPI Lifespan**: The lifespan context manager is the correct place for database initialization in FastAPI apps
4. **SocketIO Wrapping**: Even though the app is wrapped in `socketio.ASGIApp`, FastAPI's lifespan still executes properly

## Testing

To verify the fix works:
```bash
cd /Users/wuy/Desktop/code/UAgent
bash start_openhands_research.sh

# In another terminal:
curl --noproxy "*" http://localhost:2999/api/research/health
curl --noproxy "*" http://localhost:2999/api/research/experiments
```

Both endpoints should return valid JSON without any "Database not initialized" errors.
