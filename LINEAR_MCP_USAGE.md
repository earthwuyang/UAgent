# Linear MCP Usage Guide

## Overview

The Linear MCP (Model Context Protocol) server provides a standardized interface that allows AI models and agents to access Linear data securely. This guide covers setup, configuration, and usage.

## Key Information

- **Official Documentation**: https://linear.app/docs/mcp
- **HTTP Endpoint**: `https://mcp.linear.app/mcp`
- **SSE Endpoint**: `https://mcp.linear.app/sse` (recommended)
- **Authentication**: OAuth 2.1 with dynamic client registration
- **Transport Protocols**: Server-Sent Events (SSE) and Streamable HTTP

## Team Information

- **Team Name**: Uagent-ai
- **Team ID**: `58e733ae-1c0d-408b-a023-465018629a40`

## Setup Instructions

### Claude Desktop (macOS)

1. Edit the configuration file:
   ```bash
   nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Add the following configuration:
   ```json
   {
     "mcpServers": {
       "linear": {
         "command": "npx",
         "args": ["-y", "mcp-remote", "https://mcp.linear.app/sse"]
       }
     }
   }
   ```

3. Restart Claude Desktop

### Claude Code

```bash
claude mcp add --transport sse linear-server https://mcp.linear.app/sse
```

Then run `/mcp` in a Claude Code session to authenticate.

### Cursor IDE

- **Direct Installation**: [Install Link](cursor://anysphere.cursor-deeplink/mcp/install?name=Linear&config=eyJ1cmwiOiJodHRwczovL21jcC5saW5lYXIuYXBwL3NzZSJ9)
- **Manual Setup**: See [Cursor MCP Tools page](https://docs.cursor.com/en/tools/mcp)

### Visual Studio Code

1. Press `Ctrl/Cmd + P` and search for "MCP: Add Server"
2. Select "Command (stdio)"
3. Enter: `npx mcp-remote https://mcp.linear.app/sse`
4. Name it "Linear"
5. Activate via "MCP: List Servers" → Linear → Start Server

### Other Clients Configuration

For other MCP-compatible tools:
- **Command**: `npx`
- **Arguments**: `"-y", "mcp-remote", "https://mcp.linear.app/sse"`
- **Environment**: None

## Available MCP Tools

Based on the available Linear MCP tools in the current environment:

### Issue Management
- `create_issue`: Create new Linear issues
- `update_issue`: Update existing issues
- `get_issue`: Retrieve issue details
- `list_issues`: List and filter issues

### Project Management
- `create_project`: Create new projects
- `update_project`: Update existing projects
- `get_project`: Retrieve project details
- `list_projects`: List and filter projects

### Team & User Management
- `get_team`: Retrieve team information
- `list_teams`: List available teams
- `get_user`: Get user details
- `list_users`: List workspace users

### Comments & Communication
- `create_comment`: Add comments to issues
- `list_comments`: List issue comments

### Organization
- `list_cycles`: Retrieve team cycles
- `list_issue_labels`: List available labels
- `list_issue_statuses`: List issue statuses
- `create_issue_label`: Create new labels

### Documentation
- `get_document`: Retrieve Linear documents
- `list_documents`: List workspace documents

## Common Usage Patterns

### Creating an Issue

```json
{
  "title": "Issue title",
  "team": "Uagent-ai",  // or team ID: "58e733ae-1c0d-408b-a023-465018629a40"
  "description": "Issue description in Markdown",
  "assignee": "me",  // or user ID/email
  "labels": ["bug", "high-priority"],
  "priority": 1  // 0=No priority, 1=Urgent, 2=High, 3=Normal, 4=Low
}
```

### Updating an Issue

```json
{
  "id": "issue-id",
  "title": "Updated title",
  "state": "In Progress",
  "assignee": "user@example.com"
}
```

### Listing Issues

```json
{
  "team": "Uagent-ai",
  "assignee": "me",
  "state": "open",
  "limit": 50
}
```

### Creating a Project

```json
{
  "name": "Project Name",
  "team": "Uagent-ai",
  "description": "Project description in Markdown",
  "summary": "Brief summary",
  "lead": "me"
}
```

## Authentication Flow

1. First MCP call will trigger OAuth 2.1 authentication
2. You'll be redirected to Linear's authorization page
3. Grant necessary permissions
4. Authentication token is stored automatically
5. Subsequent calls use the stored token

## Best Practices

1. **Use SSE endpoint** (`https://mcp.linear.app/sse`) for better reliability
2. **Specify team context** - Always include team name or ID
3. **Use meaningful labels** - Leverage Linear's labeling system
4. **Set appropriate priorities** - Use priority levels consistently
5. **Link related items** - Connect issues to projects when relevant

## Troubleshooting

### Common Issues

1. **Internal Server Error**: Check OAuth authentication status
2. **Permission Denied**: Verify team access and user permissions
3. **WSL on Windows**: May require additional configuration
4. **Invalid Team**: Ensure team name/ID is correct

### Debug Commands

```bash
# Check MCP server status
curl -I https://mcp.linear.app/sse

# Verify team information
# Use list_teams MCP tool to confirm team details
```

## Rate Limits & Considerations

- Follow Linear's API rate limits
- Use batch operations when possible
- Cache frequently accessed data locally
- Implement proper error handling

## Advanced Features

### Custom Fields
- Use Linear's custom fields in issue creation/updates
- Reference field IDs for specific team configurations

### Automation
- Integrate with CI/CD pipelines
- Set up automated issue creation from monitoring systems
- Link issues to code commits and deployments

### Filtering & Search
- Use advanced filtering options in list operations
- Implement search functionality for better issue discovery
- Leverage Linear's query syntax for complex searches

## Security Notes

- OAuth tokens are managed automatically
- No need to store API keys manually
- Permissions are scoped to your Linear workspace access
- Revoke access through Linear's security settings if needed

## References

- [Linear MCP Official Docs](https://linear.app/docs/mcp)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification/2025-03-26)
- [mcp-remote GitHub Repository](https://github.com/geelen/mcp-remote)

---

*Last updated: 2025-10-11*
*Environment: UAgent project, macOS*