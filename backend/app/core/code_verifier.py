"""
Code Verification System for UAgent
Tests and verifies generated code before claiming success
"""

import os
import subprocess
import tempfile
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)


class CodeVerificationResult:
    """Result of code verification"""

    def __init__(self, success: bool, errors: List[str] = None, warnings: List[str] = None,
                 suggestions: List[str] = None, execution_output: str = ""):
        self.success = success
        self.errors = errors or []
        self.warnings = warnings or []
        self.suggestions = suggestions or []
        self.execution_output = execution_output


class CodeVerifier:
    """Verifies generated code and deployment scripts"""

    def __init__(self):
        self.verification_timeout = 30  # seconds

    def verify_infrastructure_deployment(self, workspace_path: str, service_type: str) -> CodeVerificationResult:
        """Verify infrastructure deployment scripts work correctly"""
        logger.info(f"🔍 Verifying {service_type} deployment in {workspace_path}")

        workspace = Path(workspace_path)
        errors = []
        warnings = []
        suggestions = []
        output_lines = []

        try:
            # Check required files exist
            required_files = ["docker-compose.yml", "deploy.sh", "manage.sh", "test.sh"]
            missing_files = []
            for file in required_files:
                if not (workspace / file).exists():
                    missing_files.append(file)

            if missing_files:
                errors.append(f"Missing required files: {', '.join(missing_files)}")
                return CodeVerificationResult(False, errors, warnings, suggestions)

            # Verify docker-compose.yml syntax and content
            compose_result = self._verify_docker_compose(workspace / "docker-compose.yml", service_type)
            if not compose_result.success:
                errors.extend(compose_result.errors)
                warnings.extend(compose_result.warnings)
                suggestions.extend(compose_result.suggestions)

            # Verify shell scripts syntax
            for script in ["deploy.sh", "manage.sh", "test.sh"]:
                script_result = self._verify_shell_script(workspace / script)
                if not script_result.success:
                    errors.extend([f"{script}: {err}" for err in script_result.errors])
                warnings.extend([f"{script}: {warn}" for warn in script_result.warnings])

            # Test Docker image availability
            image_result = self._test_docker_image_availability(workspace / "docker-compose.yml")
            if not image_result.success:
                errors.extend(image_result.errors)
                suggestions.extend(image_result.suggestions)

            # Test deployment dry-run if no errors so far
            if not errors:
                deploy_result = self._test_deployment_dry_run(workspace, service_type)
                if not deploy_result.success:
                    errors.extend(deploy_result.errors)
                    suggestions.extend(deploy_result.suggestions)
                output_lines.append(deploy_result.execution_output)

            success = len(errors) == 0

            if success:
                logger.info(f"✅ {service_type} deployment verification passed")
            else:
                logger.warning(f"❌ {service_type} deployment verification failed: {errors}")

            return CodeVerificationResult(
                success=success,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                execution_output="\n".join(output_lines)
            )

        except Exception as e:
            logger.error(f"Verification process failed: {e}")
            return CodeVerificationResult(
                False,
                [f"Verification process error: {str(e)}"],
                execution_output=str(e)
            )

    def _verify_docker_compose(self, compose_file: Path, service_type: str) -> CodeVerificationResult:
        """Verify docker-compose.yml file"""
        errors = []
        warnings = []
        suggestions = []

        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            # Check basic structure
            if 'services' not in compose_data:
                errors.append("docker-compose.yml missing 'services' section")
                return CodeVerificationResult(False, errors)

            services = compose_data['services']

            # Service-specific validation
            if service_type.lower() == 'mongodb':
                mongo_result = self._verify_mongodb_compose(services)
                errors.extend(mongo_result.errors)
                warnings.extend(mongo_result.warnings)
                suggestions.extend(mongo_result.suggestions)
            elif service_type.lower() == 'postgresql':
                postgres_result = self._verify_postgresql_compose(services)
                errors.extend(postgres_result.errors)
                warnings.extend(postgres_result.warnings)
                suggestions.extend(postgres_result.suggestions)
            elif service_type.lower() == 'redis':
                redis_result = self._verify_redis_compose(services)
                errors.extend(redis_result.errors)
                warnings.extend(redis_result.warnings)
                suggestions.extend(redis_result.suggestions)

            # Test YAML syntax with docker-compose
            result = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "config"],
                cwd=compose_file.parent,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                errors.append(f"docker-compose config validation failed: {result.stderr}")

        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML syntax: {str(e)}")
        except subprocess.TimeoutExpired:
            errors.append("docker-compose config validation timed out")
        except FileNotFoundError:
            warnings.append("docker-compose command not available for validation")
        except Exception as e:
            errors.append(f"docker-compose validation error: {str(e)}")

        return CodeVerificationResult(len(errors) == 0, errors, warnings, suggestions)

    def _verify_mongodb_compose(self, services: Dict) -> CodeVerificationResult:
        """Verify MongoDB-specific Docker Compose configuration"""
        errors = []
        warnings = []
        suggestions = []

        # Look for MongoDB service
        mongo_service = None
        for service_name, config in services.items():
            if 'mongo' in service_name.lower() or 'mongodb' in service_name.lower():
                mongo_service = config
                break

        if not mongo_service:
            errors.append("No MongoDB service found in docker-compose.yml")
            return CodeVerificationResult(False, errors)

        # Check image
        image = mongo_service.get('image', '')
        if image == 'mongodb:latest':
            errors.append("Invalid Docker image 'mongodb:latest' - should be 'mongo:6.0' or similar")
            suggestions.append("Use 'mongo:6.0' or 'mongo:latest' as the Docker image")
        elif not image.startswith('mongo'):
            warnings.append(f"Unusual MongoDB image: {image}")

        # Check ports
        ports = mongo_service.get('ports', [])
        if not ports:
            warnings.append("No ports exposed for MongoDB")
        else:
            has_mongo_port = any('27017' in str(port) for port in ports)
            if not has_mongo_port:
                warnings.append("MongoDB default port 27017 not exposed")

        # Check volumes for data persistence
        volumes = mongo_service.get('volumes', [])
        if not volumes:
            warnings.append("No volumes configured - data will not persist")
            suggestions.append("Add volume mapping for /data/db to persist MongoDB data")

        return CodeVerificationResult(len(errors) == 0, errors, warnings, suggestions)

    def _verify_postgresql_compose(self, services: Dict) -> CodeVerificationResult:
        """Verify PostgreSQL-specific Docker Compose configuration"""
        errors = []
        warnings = []
        suggestions = []

        # Look for PostgreSQL service
        postgres_service = None
        for service_name, config in services.items():
            if 'postgres' in service_name.lower():
                postgres_service = config
                break

        if not postgres_service:
            errors.append("No PostgreSQL service found in docker-compose.yml")
            return CodeVerificationResult(False, errors)

        # Check image
        image = postgres_service.get('image', '')
        if not image.startswith('postgres'):
            errors.append(f"Invalid PostgreSQL image: {image}")
            suggestions.append("Use 'postgres:15' or 'postgres:latest'")

        # Check environment variables
        env = postgres_service.get('environment', {})
        if isinstance(env, list):
            env = {item.split('=')[0]: item.split('=')[1] for item in env if '=' in item}

        required_env = ['POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_env = [var for var in required_env if var not in env]
        if missing_env:
            errors.append(f"Missing PostgreSQL environment variables: {missing_env}")

        return CodeVerificationResult(len(errors) == 0, errors, warnings, suggestions)

    def _verify_redis_compose(self, services: Dict) -> CodeVerificationResult:
        """Verify Redis-specific Docker Compose configuration"""
        errors = []
        warnings = []
        suggestions = []

        # Look for Redis service
        redis_service = None
        for service_name, config in services.items():
            if 'redis' in service_name.lower():
                redis_service = config
                break

        if not redis_service:
            errors.append("No Redis service found in docker-compose.yml")
            return CodeVerificationResult(False, errors)

        # Check image
        image = redis_service.get('image', '')
        if not image.startswith('redis'):
            errors.append(f"Invalid Redis image: {image}")
            suggestions.append("Use 'redis:7' or 'redis:latest'")

        return CodeVerificationResult(len(errors) == 0, errors, warnings, suggestions)

    def _verify_shell_script(self, script_path: Path) -> CodeVerificationResult:
        """Verify shell script syntax"""
        errors = []
        warnings = []

        try:
            # Check if file is executable
            if not os.access(script_path, os.X_OK):
                warnings.append(f"{script_path.name} is not executable")

            # Basic syntax check with bash -n
            result = subprocess.run(
                ["bash", "-n", str(script_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                errors.append(f"Syntax error: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            errors.append("Script syntax check timed out")
        except Exception as e:
            errors.append(f"Script verification error: {str(e)}")

        return CodeVerificationResult(len(errors) == 0, errors, warnings)

    def _test_docker_image_availability(self, compose_file: Path) -> CodeVerificationResult:
        """Test if Docker images specified in compose file are available"""
        errors = []
        suggestions = []

        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            services = compose_data.get('services', {})
            for service_name, config in services.items():
                image = config.get('image', '')
                if image:
                    # Test if image can be pulled
                    result = subprocess.run(
                        ["docker", "image", "inspect", image],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode != 0:
                        # Try to pull the image
                        pull_result = subprocess.run(
                            ["docker", "pull", image],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )

                        if pull_result.returncode != 0:
                            errors.append(f"Cannot pull Docker image '{image}': {pull_result.stderr.strip()}")
                            if 'mongodb:latest' in image:
                                suggestions.append("Use 'mongo:6.0' instead of 'mongodb:latest'")
                            elif 'postgresql:latest' in image:
                                suggestions.append("Use 'postgres:15' instead of 'postgresql:latest'")

        except Exception as e:
            errors.append(f"Docker image check failed: {str(e)}")

        return CodeVerificationResult(len(errors) == 0, errors, [], suggestions)

    def _test_deployment_dry_run(self, workspace: Path, service_type: str) -> CodeVerificationResult:
        """Test deployment with dry-run to catch issues"""
        errors = []
        suggestions = []
        output_lines = []

        try:
            # Test docker-compose up in dry-run mode
            result = subprocess.run(
                ["docker-compose", "up", "--dry-run"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.verification_timeout
            )

            output_lines.append(f"Docker Compose dry-run output:\n{result.stdout}")

            if result.returncode != 0:
                errors.append(f"docker-compose up dry-run failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            errors.append("Deployment dry-run timed out")
        except FileNotFoundError:
            suggestions.append("docker-compose command not available for testing")
        except Exception as e:
            errors.append(f"Deployment test error: {str(e)}")

        return CodeVerificationResult(
            len(errors) == 0,
            errors,
            [],
            suggestions,
            "\n".join(output_lines)
        )

    def verify_general_code(self, workspace_path: str, task_description: str) -> CodeVerificationResult:
        """Verify general code (Python scripts, programs, etc.)"""
        logger.info(f"🔍 Verifying general code in {workspace_path}")

        workspace = Path(workspace_path)
        errors = []
        warnings = []
        suggestions = []
        output_lines = []

        try:
            # Check if workspace exists
            if not workspace.exists():
                errors.append(f"Workspace directory does not exist: {workspace_path}")
                return CodeVerificationResult(False, errors, warnings, suggestions)

            # Find Python files to verify
            python_files = list(workspace.glob("*.py"))
            shell_scripts = list(workspace.glob("*.sh"))

            # Verify Python files
            for py_file in python_files:
                result = self._verify_python_file(py_file)
                if not result.success:
                    errors.extend([f"{py_file.name}: {err}" for err in result.errors])
                warnings.extend([f"{py_file.name}: {warn}" for warn in result.warnings])

            # Verify shell scripts
            for script in shell_scripts:
                result = self._verify_shell_script(script)
                if not result.success:
                    errors.extend([f"{script.name}: {err}" for err in result.errors])
                warnings.extend([f"{script.name}: {warn}" for warn in result.warnings])

            # Try to execute Python files if they look like main scripts
            main_files = [f for f in python_files if f.name in ['main.py', 'app.py', 'run.py']]
            if main_files:
                for main_file in main_files:
                    exec_result = self._test_python_execution(main_file)
                    if not exec_result.success:
                        errors.extend(exec_result.errors)
                        suggestions.extend(exec_result.suggestions)
                    output_lines.append(exec_result.execution_output)

            success = len(errors) == 0

            if success:
                logger.info(f"✅ General code verification passed")
            else:
                logger.warning(f"❌ General code verification failed: {errors}")

            return CodeVerificationResult(
                success=success,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                execution_output="\n".join(output_lines)
            )

        except Exception as e:
            logger.error(f"General code verification failed: {e}")
            return CodeVerificationResult(
                False,
                [f"Verification process error: {str(e)}"],
                execution_output=str(e)
            )

    def _verify_python_file(self, py_file: Path) -> CodeVerificationResult:
        """Verify Python file syntax and basic structure"""
        errors = []
        warnings = []

        try:
            # Check syntax
            with open(py_file, 'r') as f:
                source = f.read()

            compile(source, str(py_file), 'exec')

            # Check for common issues
            if len(source.strip()) == 0:
                warnings.append("File is empty")
            elif len(source.split('\n')) < 3:
                warnings.append("File seems very short")

            # Check for basic Python structure
            if '__name__ == "__main__"' not in source and py_file.name == 'main.py':
                suggestions = ["Consider adding if __name__ == '__main__': block"]
            else:
                suggestions = []

        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
            suggestions = ["Fix Python syntax errors"]
        except Exception as e:
            errors.append(f"Python file verification error: {str(e)}")
            suggestions = []

        return CodeVerificationResult(len(errors) == 0, errors, warnings, suggestions)

    def _test_python_execution(self, py_file: Path) -> CodeVerificationResult:
        """Test Python file execution"""
        errors = []
        suggestions = []
        output = ""

        try:
            # Try to run the Python file with a timeout
            result = subprocess.run(
                ["python3", str(py_file)],
                cwd=py_file.parent,
                capture_output=True,
                text=True,
                timeout=10
            )

            output = f"stdout: {result.stdout}\nstderr: {result.stderr}"

            if result.returncode != 0:
                errors.append(f"Python execution failed with return code {result.returncode}")
                if result.stderr:
                    errors.append(f"Error: {result.stderr.strip()}")
                suggestions.append("Fix runtime errors in Python code")
            else:
                logger.info(f"✅ Python file {py_file.name} executed successfully")

        except subprocess.TimeoutExpired:
            errors.append("Python execution timed out (>10s)")
            suggestions.append("Optimize code performance or remove infinite loops")
        except Exception as e:
            errors.append(f"Python execution test failed: {str(e)}")

        return CodeVerificationResult(
            len(errors) == 0,
            errors,
            [],
            suggestions,
            output
        )

    def generate_fix_suggestions(self, verification_result: CodeVerificationResult,
                               service_type: str) -> List[str]:
        """Generate specific fix suggestions based on verification results"""
        fixes = []

        for error in verification_result.errors:
            if "mongodb:latest" in error:
                fixes.append("Change 'mongodb:latest' to 'mongo:6.0' in docker-compose.yml")
            elif "postgresql:latest" in error:
                fixes.append("Change 'postgresql:latest' to 'postgres:15' in docker-compose.yml")
            elif "Missing required files" in error:
                fixes.append("Generate all required deployment files: docker-compose.yml, deploy.sh, manage.sh, test.sh")
            elif "POSTGRES_" in error:
                fixes.append("Add required PostgreSQL environment variables: POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")

        fixes.extend(verification_result.suggestions)

        return fixes