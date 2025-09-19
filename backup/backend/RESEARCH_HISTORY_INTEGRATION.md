# Research History Integration Guide

This guide explains how to integrate the research history sidebar into your existing frontend application.

## Overview

The research history system provides:
- SQLite database persistence for all research goals, nodes, and results
- REST API endpoints for history management
- Pre-built HTML/CSS/JS sidebar component
- Real-time updates via WebSocket connections
- Workspace management with relative project paths (outside backend directory)

## Backend Integration Status

✅ **Already Completed:**
- Database schema created (`data/research_history.db`)
- `DatabaseIntegratedResearchSystem` initialized in `/app/routers/research_tree.py:24`
- API endpoints added to research tree router
- All data is automatically persisted to database

## Frontend Integration Steps

### 1. Include the Sidebar HTML

The sidebar template is located at:
```
/frontend_templates/research_history_sidebar.html
```

**Option A: Direct Inclusion**
```html
<!-- In your main layout file -->
<div id="app-container">
    <!-- Include the sidebar -->
    <!-- Copy content from research_history_sidebar.html -->
    <div id="research-history-sidebar" class="sidebar">
        <!-- Sidebar content here -->
    </div>

    <!-- Your main content -->
    <div id="main-content">
        <!-- Existing application content -->
    </div>
</div>
```

**Option B: Dynamic Loading**
```javascript
// Load sidebar dynamically
fetch('/api/research-tree/frontend/sidebar')
    .then(response => response.text())
    .then(html => {
        document.body.insertAdjacentHTML('afterbegin', html);
        initializeResearchHistorySidebar();
    });
```

### 2. CSS Integration

The sidebar includes its own CSS with:
- Purple gradient theme
- Responsive design
- Status badges and animations

**To customize:**
```css
/* Override sidebar colors */
:root {
    --sidebar-primary: #your-color;
    --sidebar-secondary: #your-secondary;
}

/* Adjust sidebar width */
.sidebar {
    width: 350px; /* Default is 300px */
}

/* Adjust main content margin */
.main-content {
    margin-left: 350px; /* Match sidebar width */
}
```

### 3. JavaScript Integration

**Initialize the sidebar:**
```javascript
// After sidebar HTML is loaded
const historySidebar = new ResearchHistorySidebar({
    apiBaseUrl: '/api/research-tree',
    websocketUrl: 'ws://localhost:8000/api/research-tree'
});

await historySidebar.initialize();
```

**Handle sidebar events:**
```javascript
// Listen for sidebar events
document.addEventListener('research-history-goal-selected', (event) => {
    const goalId = event.detail.goalId;
    // Navigate to goal details or update main content
    window.location.href = `/goals/${goalId}`;
});

document.addEventListener('research-history-goal-archived', (event) => {
    const goalId = event.detail.goalId;
    // Show success message
    showNotification(`Goal ${goalId} archived successfully`);
});
```

### 4. API Endpoints Reference

**Available endpoints:**
```javascript
// Get research history (with optional filters)
GET /api/research-tree/history?limit=50&status=active

// Get goal details
GET /api/research-tree/goals/{goal_id}/details

// Archive goal
POST /api/research-tree/goals/{goal_id}/archive

// Restore goal
POST /api/research-tree/goals/{goal_id}/restore

// Delete goal
DELETE /api/research-tree/goals/{goal_id}

// Search goals
GET /api/research-tree/history/search?q=docker&limit=20

// Get database stats
GET /api/research-tree/stats
```

**WebSocket endpoints:**
```javascript
// Real-time goal updates
WS /api/research-tree/goals/{goal_id}/logs

// Global research updates
WS /api/research-tree/updates
```

### 5. Example Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>Research System</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        .app-container { display: flex; min-height: 100vh; }
        .main-content { flex: 1; padding: 20px; }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Research History Sidebar -->
        <div id="research-history-sidebar"></div>

        <!-- Main Application Content -->
        <div class="main-content">
            <h1>Research System</h1>
            <div id="current-goal-content">
                <!-- Your existing application -->
            </div>
        </div>
    </div>

    <script>
        // Load and initialize sidebar
        async function initializeApp() {
            // Load sidebar HTML
            const response = await fetch('/frontend_templates/research_history_sidebar.html');
            const sidebarHtml = await response.text();
            document.getElementById('research-history-sidebar').innerHTML = sidebarHtml;

            // Initialize sidebar functionality
            const sidebar = new ResearchHistorySidebar();
            await sidebar.initialize();

            // Handle goal selection
            document.addEventListener('research-history-goal-selected', (event) => {
                loadGoalDetails(event.detail.goalId);
            });
        }

        function loadGoalDetails(goalId) {
            // Load goal details into main content
            fetch(`/api/research-tree/goals/${goalId}/details`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('current-goal-content').innerHTML =
                        `<h2>${data.title}</h2><p>${data.description}</p>`;
                });
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeApp);
    </script>
</body>
</html>
```

## Configuration Options

### Database Configuration

The system uses SQLite by default with database path: `data/research_history.db`

**To change database path:**
```python
# In app/routers/research_tree.py
research_system = DatabaseIntegratedResearchSystem(
    db_path="custom/path/research.db"
)
```

### API Configuration

**CORS Settings:**
Ensure your CORS settings allow frontend access:
```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Common Issues

1. **Database not found:**
   - Ensure `data/` directory exists
   - Check database path in `DatabaseIntegratedResearchSystem`

2. **WebSocket connection fails:**
   - Verify WebSocket URL matches your server
   - Check CORS settings for WebSocket connections

3. **Sidebar not loading:**
   - Check browser console for JavaScript errors
   - Verify API endpoints are accessible

4. **History not updating:**
   - Check database writes are working
   - Verify WebSocket events are being sent

### Database Inspection

```bash
# View database contents
sqlite3 data/research_history.db

# List all tables
.tables

# View research goals
SELECT id, title, status, created_at FROM research_goals;

# View goal statistics
SELECT status, COUNT(*) FROM research_goals GROUP BY status;
```

## Next Steps

1. Copy the sidebar HTML template to your frontend
2. Add the CSS and JavaScript integration
3. Test with a simple goal creation
4. Customize styling to match your application theme
5. Add error handling and user feedback

The system is fully functional and ready for integration!