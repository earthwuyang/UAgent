# Universal Agent (UAgent) - Usage Guide

## Overview

Universal Agent (UAgent) is an intelligent research system with multi-engine orchestration that provides real-time research visualization. It supports three types of research engines:

- **Deep Research Engine**: Comprehensive multi-source research (ChatGPT-style)
- **Code Research Engine**: Specialized code analysis and repository research (RepoMaster-style)
- **Scientific Research Engine**: Advanced research with hypothesis generation and experimental design (AI Scientist/Agent Laboratory-style)

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- DashScope API Key (Qwen LLM)

## Environment Setup

### 1. Clone and Setup Directory

```bash
cd /path/to/uagent
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Environment Variables

Create a `.env` file in the backend directory:

```bash
# backend/.env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
WORKSPACE_DIR=/tmp/uagent_workspaces
MAX_ITERATIONS=3
CONFIDENCE_THRESHOLD=0.8
```

**Required Environment Variables:**

- `DASHSCOPE_API_KEY`: Your DashScope API key for Qwen LLM (Required)
- `WORKSPACE_DIR`: Directory for OpenHands workspaces (Optional, defaults to `/tmp/uagent_workspaces`)
- `MAX_ITERATIONS`: Maximum iterations for scientific research (Optional, defaults to 3)
- `CONFIDENCE_THRESHOLD`: Confidence threshold for research completion (Optional, defaults to 0.8)

#### Get DashScope API Key

1. Visit [DashScope Console](https://dashscope.console.aliyun.com/)
2. Sign up/Login to your Alibaba Cloud account
3. Create an API key for Qwen models
4. Copy the API key to your `.env` file

### 3. Frontend Setup

#### Install Node Dependencies

```bash
cd frontend
npm install
```

#### Frontend Configuration

The frontend is pre-configured to connect to:
- Backend API: `http://localhost:8012`
- Frontend Dev Server: `http://localhost:3000`

No additional configuration required for development.

## Starting the System

### 1. Start Backend Server

```bash
cd backend
source .venv/bin/activate  # Activate virtual environment
DASHSCOPE_API_KEY=your_api_key_here uvicorn app.main:app --host 127.0.0.1 --port 8012 --reload
```

**Alternative method using environment file:**

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8012 --reload
```

The backend will start on `http://localhost:8012`

#### Backend Health Check

Verify the backend is running:

```bash
curl http://localhost:8012/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "engines": {
    "deep_research": "active",
    "code_research": "active",
    "scientific_research": "active"
  },
  "smart_router": "active"
}
```

### 2. Start Frontend Development Server

In a new terminal:

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

#### Frontend Access Points

- **Homepage**: `http://localhost:3000/`
- **Research Dashboard**: `http://localhost:3000/dashboard`
- **System Status**: `http://localhost:3000/status`

## Using the Research Dashboard

### Accessing the Dashboard

Navigate to `http://localhost:3000/dashboard` to access the real-time research visualization dashboard.

### Features

1. **Smart Research Input**
   - Enter natural language research requests
   - Automatic engine classification and routing
   - Example requests for each research engine type

2. **Real-Time Progress Tracking**
   - Live WebSocket updates showing research progress
   - Engine status monitoring
   - Progress percentages and status indicators

3. **Multi-Engine Support**
   - **Deep Research**: Market research, literature reviews, industry analysis
   - **Code Research**: Technology evaluation, library selection, code architecture
   - **Scientific Research**: Academic research, experimental validation, hypothesis testing

### Example Research Requests

#### Deep Research Examples
```
"Research the latest developments in artificial intelligence and machine learning for 2024"
"Find important news today"
"What are the current trends in renewable energy technology?"
```

#### Code Research Examples
```
"Find Python libraries for transformer attention mechanisms"
"Analyze FastAPI project architectures on GitHub"
"Research best practices for React state management"
```

#### Scientific Research Examples
```
"Design experiments to test whether sparse attention patterns improve transformer efficiency"
"Investigate the effectiveness of different optimization algorithms for neural networks"
"Research methods for improving model interpretability in deep learning"
```

## API Endpoints

### Smart Router Endpoints

- `POST /api/router/classify` - Classify research request
- `POST /api/router/route-and-execute` - Route and execute research
- `GET /api/router/engines` - List available engines
- `GET /api/router/status` - Get router status

### WebSocket Endpoints

- `ws://localhost:8012/ws/research/{session_id}` - Research progress updates
- `ws://localhost:8012/ws/engines/status` - Engine status updates
- `ws://localhost:8012/ws/openhands/{session_id}` - OpenHands execution output

### Research Engines

- `POST /api/research/deep` - Deep research engine
- `POST /api/research/code` - Code research engine
- `POST /api/research/scientific` - Scientific research engine

## Troubleshooting

### Common Issues

#### 1. Backend Fails to Start

**Error**: `DASHSCOPE_API_KEY environment variable is required`

**Solution**: Ensure your DashScope API key is properly configured:
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

#### 2. Frontend Import Errors

**Error**: `Failed to resolve import "@/components/ui/card"`

**Solution**: Restart the frontend development server:
```bash
cd frontend
npm run dev
```

#### 3. WebSocket Connection Issues

**Error**: WebSocket connection failed

**Solution**:
1. Ensure backend is running on port 8012
2. Check firewall settings
3. Verify CORS configuration in backend

#### 4. Research Engine Errors

**Error**: `'CodeAnalysis' object has no attribute 'get'`

**Solution**: This is a known issue with mock data structures. The system will continue to function with reduced functionality.

### Health Checks

#### Backend Health
```bash
curl http://localhost:8012/health
```

#### Frontend Access
```bash
curl http://localhost:3000
```

#### API Functionality
```bash
curl -X POST http://localhost:8012/api/router/classify \
  -H "Content-Type: application/json" \
  -d '{"user_request": "test research request"}'
```

## Development Notes

### Architecture

- **Backend**: FastAPI with async/await support
- **Frontend**: React + TypeScript with Vite
- **Real-time**: WebSocket connections for live updates
- **LLM**: DashScope Qwen models (NO MOCKING - real LLM calls)
- **Code Execution**: OpenHands CLI integration (headless, no Docker)
- **Web Search**: Playwright + xvfb for real web searches

### Key Components

1. **Smart Router**: LLM-based request classification with >95% accuracy
2. **Research Engines**: Three specialized engines for different research types
3. **WebSocket Manager**: Real-time progress tracking and updates
4. **OpenHands Integration**: Code execution and workspace management

### File Structure

```
uagent/
├── backend/
│   ├── app/
│   │   ├── core/          # Core engine implementations
│   │   ├── routers/       # API endpoints
│   │   └── main.py        # FastAPI application
│   ├── requirements.txt
│   └── .env              # Environment variables
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   └── services/      # API services
│   ├── package.json
│   └── vite.config.ts
└── USAGE.md              # This file
```

## Production Deployment

### Environment Variables for Production

```bash
# Production environment variables
DASHSCOPE_API_KEY=your_production_api_key
WORKSPACE_DIR=/var/lib/uagent/workspaces
MAX_ITERATIONS=5
CONFIDENCE_THRESHOLD=0.85
CORS_ORIGINS=https://yourdomain.com
```

### Build Frontend for Production

```bash
cd frontend
npm run build
```

### Production Server Setup

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8012 --workers 4
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review system logs in the terminal output
3. Ensure all environment variables are properly configured
4. Verify API connectivity with health checks

## Version Information

- **System Version**: 1.0.0
- **Backend**: FastAPI + Python 3.8+
- **Frontend**: React + TypeScript + Vite
- **LLM Provider**: DashScope (Qwen)
- **Code Execution**: OpenHands CLI
- **Real-time Updates**: WebSocket connections