"""
Infrastructure Deployment Handler
Generates deployment scripts and configuration files for infrastructure requests
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .workspace_manager import WorkspaceManager
from .llm_client import llm_client


@dataclass
class DeploymentArtifact:
    """Represents a deployment artifact (script, config, etc.)"""
    name: str
    content: str
    file_type: str  # script, config, dockerfile, compose
    executable: bool = False
    description: str = ""


class InfrastructureDeployer:
    """Handles generation of infrastructure deployment scripts"""

    def __init__(self, workspace_manager: WorkspaceManager = None):
        if workspace_manager is None:
            # Always use the project workspace directory: /home/wuy/AI/uagent/workspace
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            workspace_base = os.path.join(project_root, "workspace")
            workspace_manager = WorkspaceManager(workspace_base)
        self.workspace_manager = workspace_manager
        # Remove hardcoded templates - use AI to generate everything

    def detect_infrastructure_request(self, title: str, description: str, success_criteria: List[str]) -> Optional[str]:
        """Detect if this is an infrastructure deployment request"""
        combined_text = f"{title} {description} {' '.join(success_criteria)}".lower()

        # Infrastructure keywords
        infra_keywords = [
            'setup', 'deploy', 'install', 'build', 'configure', 'running instance',
            'database server', 'web server', 'application server'
        ]

        # Database/service keywords
        service_keywords = ['postgresql', 'postgres', 'mysql', 'redis', 'mongodb',
                          'elasticsearch', 'nginx', 'apache', 'docker']

        # Check for infrastructure patterns
        has_infra_keyword = any(keyword in combined_text for keyword in infra_keywords)
        detected_service = None

        for service in service_keywords:
            if service in combined_text:
                detected_service = service
                break

        if has_infra_keyword and detected_service:
            return detected_service

        return None

    def generate_deployment(self, service_type: str, title: str, description: str,
                          success_criteria: List[str], constraints: Dict[str, Any] = None) -> str:
        """Generate deployment artifacts for the specified service"""

        # Create workspace for this deployment
        workspace_name = f"{service_type}-deployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        workspace_info = self.workspace_manager.create_workspace(
            workspace_name,
            f"Infrastructure deployment for {title}"
        )

        # Extract path from workspace info
        if isinstance(workspace_info, dict):
            workspace_path = workspace_info.get('path', workspace_info.get('workspace_path'))
        else:
            workspace_path = workspace_info

        # Ensure workspace path exists
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path, exist_ok=True)

        # Trigger UAgent research to generate deployment artifacts
        # This will be handled by the research system creating a workspace task
        return workspace_path  # Return early, let research system handle generation

    def _generate_postgresql_deployment(self, title: str, description: str,
                                      success_criteria: List[str], constraints: Dict[str, Any]) -> List[DeploymentArtifact]:
        """Generate PostgreSQL deployment artifacts"""

        # Extract configuration from constraints
        db_name = constraints.get('database_name', 'research_db')
        db_user = constraints.get('username', 'postgres')
        db_password = constraints.get('password', 'secure_password_123')
        port = constraints.get('port', 5432)
        version = constraints.get('postgres_version', '15')

        artifacts = []

        # Docker Compose file
        compose_content = f"""version: '3.8'

services:
  postgres:
    image: postgres:{version}
    container_name: postgresql_research
    restart: unless-stopped
    environment:
      POSTGRES_DB: {db_name}
      POSTGRES_USER: {db_user}
      POSTGRES_PASSWORD: {db_password}
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "{port}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    networks:
      - postgres_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U {db_user} -d {db_name}"]
      interval: 10s
      timeout: 5s
      retries: 5

  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: pgadmin_research
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@research.local
      PGADMIN_DEFAULT_PASSWORD: admin123
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    ports:
      - "8080:80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    networks:
      - postgres_network
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
    driver: local
  pgadmin_data:
    driver: local

networks:
  postgres_network:
    driver: bridge
"""
        artifacts.append(DeploymentArtifact(
            name="docker-compose.yml",
            content=compose_content,
            file_type="compose",
            description="Docker Compose configuration for PostgreSQL with pgAdmin"
        ))

        # Deployment script
        deploy_script = f"""#!/bin/bash
# PostgreSQL Deployment Script
# Generated for: {title}

set -e

echo "🐘 Starting PostgreSQL deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if ports are available
if lsof -Pi :{port} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Port {port} is already in use. Please stop the service using this port."
    exit 1
fi

if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Port 8080 is already in use. Please stop the service using this port."
    exit 1
fi

# Create init scripts directory
mkdir -p init-scripts

# Deploy PostgreSQL
echo "🚀 Deploying PostgreSQL container..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
timeout 60 bash -c 'until docker-compose exec -T postgres pg_isready -U {db_user} -d {db_name}; do sleep 2; done'

if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL is ready!"
    echo ""
    echo "📊 Connection Details:"
    echo "  Host: localhost"
    echo "  Port: {port}"
    echo "  Database: {db_name}"
    echo "  Username: {db_user}"
    echo "  Password: {db_password}"
    echo ""
    echo "🌐 pgAdmin Web Interface:"
    echo "  URL: http://localhost:8080"
    echo "  Email: admin@research.local"
    echo "  Password: admin123"
    echo ""
    echo "🔧 To connect from command line:"
    echo "  psql -h localhost -p {port} -U {db_user} -d {db_name}"
    echo ""
    echo "📝 To run SQL scripts:"
    echo "  Place .sql files in ./init-scripts/ before deployment"
    echo "  Or use: docker-compose exec postgres psql -U {db_user} -d {db_name} -f /path/to/script.sql"
else
    echo "❌ PostgreSQL failed to start properly"
    echo "🔍 Check logs with: docker-compose logs postgres"
    exit 1
fi
"""
        artifacts.append(DeploymentArtifact(
            name="deploy.sh",
            content=deploy_script,
            file_type="script",
            executable=True,
            description="Main deployment script to start PostgreSQL"
        ))

        # Management script
        manage_script = f"""#!/bin/bash
# PostgreSQL Management Script

case "$1" in
    start)
        echo "🚀 Starting PostgreSQL..."
        docker-compose up -d
        ;;
    stop)
        echo "🛑 Stopping PostgreSQL..."
        docker-compose down
        ;;
    restart)
        echo "🔄 Restarting PostgreSQL..."
        docker-compose restart
        ;;
    status)
        echo "📊 PostgreSQL Status:"
        docker-compose ps
        echo ""
        echo "🔍 Health Check:"
        docker-compose exec postgres pg_isready -U {db_user} -d {db_name} || echo "❌ PostgreSQL not ready"
        ;;
    logs)
        echo "📋 PostgreSQL Logs:"
        docker-compose logs -f postgres
        ;;
    psql)
        echo "🗄️ Connecting to PostgreSQL..."
        docker-compose exec postgres psql -U {db_user} -d {db_name}
        ;;
    backup)
        backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
        echo "💾 Creating backup: $backup_file"
        docker-compose exec -T postgres pg_dump -U {db_user} -d {db_name} > "$backup_file"
        echo "✅ Backup created: $backup_file"
        ;;
    cleanup)
        echo "🧹 Cleaning up PostgreSQL..."
        docker-compose down -v
        docker volume prune -f
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "PostgreSQL Management Script"
        echo ""
        echo "Usage: $0 {{start|stop|restart|status|logs|psql|backup|cleanup}}"
        echo ""
        echo "Commands:"
        echo "  start    - Start PostgreSQL services"
        echo "  stop     - Stop PostgreSQL services"
        echo "  restart  - Restart PostgreSQL services"
        echo "  status   - Show service status and health"
        echo "  logs     - Show PostgreSQL logs"
        echo "  psql     - Connect to PostgreSQL CLI"
        echo "  backup   - Create database backup"
        echo "  cleanup  - Remove all containers and volumes"
        exit 1
        ;;
esac
"""
        artifacts.append(DeploymentArtifact(
            name="manage.sh",
            content=manage_script,
            file_type="script",
            executable=True,
            description="Management script for PostgreSQL operations"
        ))

        # Test script
        test_script = f"""#!/bin/bash
# PostgreSQL Test Script

echo "🧪 Testing PostgreSQL deployment..."

# Test connection
echo "1. Testing database connection..."
if docker-compose exec -T postgres psql -U {db_user} -d {db_name} -c "SELECT version();" > /dev/null 2>&1; then
    echo "✅ Database connection successful"
else
    echo "❌ Database connection failed"
    exit 1
fi

# Test basic operations
echo "2. Testing basic operations..."
docker-compose exec -T postgres psql -U {db_user} -d {db_name} << 'EOF'
-- Create test table
CREATE TABLE IF NOT EXISTS test_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert test data
INSERT INTO test_table (name) VALUES ('Test Record');

-- Query test data
SELECT * FROM test_table;

-- Clean up
DROP TABLE test_table;
EOF

if [ $? -eq 0 ]; then
    echo "✅ Basic operations test successful"
else
    echo "❌ Basic operations test failed"
    exit 1
fi

# Test pgAdmin accessibility
echo "3. Testing pgAdmin accessibility..."
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✅ pgAdmin web interface accessible"
else
    echo "⚠️  pgAdmin web interface not accessible (may still be starting)"
fi

echo ""
echo "🎉 PostgreSQL deployment test completed successfully!"
echo ""
echo "📊 Quick verification:"
echo "  Database: {db_name}"
echo "  Version: $(docker-compose exec -T postgres psql -U {db_user} -d {db_name} -t -c 'SELECT version();' | head -1 | xargs)"
echo "  Status: $(docker-compose exec -T postgres pg_isready -U {db_user} -d {db_name})"
"""
        artifacts.append(DeploymentArtifact(
            name="test.sh",
            content=test_script,
            file_type="script",
            executable=True,
            description="Test script to verify PostgreSQL deployment"
        ))

        # Environment configuration
        env_content = f"""# PostgreSQL Environment Configuration
# Copy to .env and modify as needed

POSTGRES_DB={db_name}
POSTGRES_USER={db_user}
POSTGRES_PASSWORD={db_password}
POSTGRES_PORT={port}
POSTGRES_VERSION={version}

PGADMIN_EMAIL=admin@research.local
PGADMIN_PASSWORD=admin123
PGADMIN_PORT=8080

# Connection URL for applications
DATABASE_URL=postgresql://{db_user}:{db_password}@localhost:{port}/{db_name}
"""
        artifacts.append(DeploymentArtifact(
            name=".env.example",
            content=env_content,
            file_type="config",
            description="Environment configuration template"
        ))

        # Sample SQL initialization script
        init_sql = f"""-- PostgreSQL Initialization Script
-- This script runs automatically when the database is first created

-- Create sample schema for research/ML work
CREATE SCHEMA IF NOT EXISTS research;

-- Create sample table for ML experiments
CREATE TABLE IF NOT EXISTS research.experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100),
    dataset_name VARCHAR(255),
    accuracy DECIMAL(5,4),
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create sample table for data storage
CREATE TABLE IF NOT EXISTS research.datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    size_mb INTEGER,
    rows_count BIGINT,
    columns_count INTEGER,
    file_path VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_experiments_name ON research.experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON research.experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON research.datasets(name);

-- Insert sample data
INSERT INTO research.datasets (name, description, rows_count, columns_count)
VALUES ('sample_dataset', 'Sample dataset for testing', 1000, 10)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA research TO {db_user};
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA research TO {db_user};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA research TO {db_user};

-- Create a user for applications (optional)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'app_password_123';
        GRANT CONNECT ON DATABASE {db_name} TO app_user;
        GRANT USAGE ON SCHEMA research TO app_user;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA research TO app_user;
        GRANT USAGE ON ALL SEQUENCES IN SCHEMA research TO app_user;
    END IF;
END
$$;

-- Log completion
INSERT INTO research.experiments (experiment_name, model_type, dataset_name, accuracy)
VALUES ('init_test', 'setup', 'sample_dataset', 1.0000);

SELECT 'PostgreSQL initialization completed successfully!' AS status;
"""
        artifacts.append(DeploymentArtifact(
            name="init-scripts/01-initialize.sql",
            content=init_sql,
            file_type="config",
            description="Database initialization script with research schema"
        ))

        return artifacts

    # Removed hardcoded AI generation - now handled by research system

    def _create_deployment_summary(self, service_type: str, title: str, workspace_path: str,
                                 created_files: List[Dict], success_criteria: List[str]) -> str:
        """Create a deployment summary markdown file"""

        # Show relative path from project root
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            relative_path = os.path.relpath(workspace_path, project_root)
            display_path = f"./{relative_path}"
        except:
            display_path = workspace_path

        summary = f"""# {service_type.upper()} Deployment Summary

**Task**: {title}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Workspace**: {display_path}
**Full Path**: {workspace_path}

## 🎯 Success Criteria

{chr(10).join(f'- {criteria}' for criteria in success_criteria)}

## 📁 Generated Files

"""

        for file_info in created_files:
            summary += f"""### {file_info['name']}
- **Type**: {file_info['type']}
- **Description**: {file_info['description']}
- **Executable**: {'Yes' if file_info['executable'] else 'No'}
- **Path**: `{file_info['path']}`

"""

        summary += f"""## 🚀 Quick Start

1. **Navigate to workspace**:
   ```bash
   cd {display_path}
   ```

2. **Deploy {service_type}**:
   ```bash
   ./deploy.sh
   ```

3. **Manage the service**:
   ```bash
   ./manage.sh status  # Check status
   ./manage.sh stop    # Stop service
   ./manage.sh start   # Start service
   ```

4. **Test the deployment**:
   ```bash
   ./test.sh
   ```

## 📝 Notes

- All deployment scripts are ready to run
- Configuration can be modified in the generated files
- Check individual file descriptions for specific usage
- Ensure Docker is running before deployment

## 🔧 Troubleshooting

If deployment fails:
1. Check Docker is running: `docker info`
2. Check port availability: `lsof -i :{5432 if service_type in ['postgresql', 'postgres'] else 'SERVICE_PORT'}`
3. View logs: `./manage.sh logs`
4. Clean up and retry: `./manage.sh cleanup && ./deploy.sh`

Generated by UAgent Infrastructure Deployer
"""

        return summary