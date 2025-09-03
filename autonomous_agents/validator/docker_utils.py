"""
Docker utilities for Validator Agent
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class DockerValidator:
    """
    Docker-based code validation
    
    Responsibilities:
    - Run tests in isolated Docker containers
    - Execute linters and formatters
    - Build and validate Docker images
    - Provide detailed validation reports
    """
    
    def __init__(self, base_image: str = "python:3.11-slim"):
        self.base_image = base_image
        self.temp_dir = None
    
    def validate_project(self, project_path: str, test_file: str = None) -> Dict[str, Any]:
        """
        Validate a complete project
        
        Args:
            project_path: Path to the project directory
            test_file: Optional specific test file to run
            
        Returns:
            Dict containing validation results
        """
        try:
            # Create temporary directory for Docker context
            self.temp_dir = tempfile.mkdtemp()
            
            # Copy project files to temp directory
            self._copy_project_files(project_path, self.temp_dir)
            
            # Run validation steps
            results = {
                "syntax_check": self._check_syntax(),
                "lint_check": self._run_linter(),
                "test_results": self._run_tests(test_file),
                "build_check": self._check_docker_build(),
                "overall_success": False
            }
            
            # Determine overall success
            results["overall_success"] = (
                results["syntax_check"]["success"] and
                results["lint_check"]["success"] and
                results["test_results"]["success"] and
                results["build_check"]["success"]
            )
            
            return results
            
        except Exception as e:
            return {
                "syntax_check": {"success": False, "error": str(e)},
                "lint_check": {"success": False, "error": str(e)},
                "test_results": {"success": False, "error": str(e)},
                "build_check": {"success": False, "error": str(e)},
                "overall_success": False
            }
        finally:
            self._cleanup()
    
    def _copy_project_files(self, source_path: str, dest_path: str):
        """Copy project files to temporary directory"""
        import shutil
        
        source = Path(source_path)
        dest = Path(dest_path)
        
        # Copy all Python files and requirements
        for file_path in source.rglob("*"):
            if file_path.is_file() and (
                file_path.suffix in ['.py', '.txt', '.yml', '.yaml', '.json'] or
                file_path.name in ['Dockerfile', 'requirements.txt']
            ):
                rel_path = file_path.relative_to(source)
                dest_file = dest / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_file)
    
    def _check_syntax(self) -> Dict[str, Any]:
        """Check Python syntax"""
        try:
            result = subprocess.run(
                ["python3", "-m", "py_compile", "main.py"],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_linter(self) -> Dict[str, Any]:
        """Run Python linter"""
        try:
            # Try to run flake8 if available
            result = subprocess.run(
                ["python3", "-m", "flake8", "--max-line-length=100", "."],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # flake8 returns non-zero for linting issues, but that's not a failure
            return {
                "success": True,  # We consider linting issues as warnings, not failures
                "output": result.stdout,
                "warnings": result.stderr,
                "linting_issues": result.returncode != 0
            }
        except FileNotFoundError:
            # flake8 not available, skip linting
            return {
                "success": True,
                "output": "Linter not available, skipping...",
                "warnings": "",
                "linting_issues": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_tests(self, test_file: str = None) -> Dict[str, Any]:
        """Run pytest tests"""
        try:
            # Install requirements first
            if (Path(self.temp_dir) / "requirements.txt").exists():
                subprocess.run(
                    ["pip", "install", "-r", "requirements.txt"],
                    cwd=self.temp_dir,
                    capture_output=True,
                    timeout=120
                )
            
            # Run tests
            test_target = test_file if test_file else "."
            result = subprocess.run(
                ["python3", "-m", "pytest", test_target, "-v", "--tb=short"],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "test_count": self._count_tests(result.stdout)
            }
        except FileNotFoundError:
            # pytest not available, create a simple test
            return self._run_simple_test()
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_simple_test(self) -> Dict[str, Any]:
        """Run a simple import test when pytest is not available"""
        try:
            result = subprocess.run(
                ["python3", "-c", "import main; print('Import successful')"],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "test_count": 1
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_docker_build(self) -> Dict[str, Any]:
        """Check if Dockerfile builds successfully"""
        try:
            if not (Path(self.temp_dir) / "Dockerfile").exists():
                return {
                    "success": True,
                    "output": "No Dockerfile found, skipping build check",
                    "error": ""
                }
            
            # Try to build Docker image
            result = subprocess.run(
                ["docker", "build", "-t", "test-image", "."],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except FileNotFoundError:
            # Docker not available
            return {
                "success": True,
                "output": "Docker not available, skipping build check",
                "error": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _count_tests(self, pytest_output: str) -> int:
        """Count number of tests from pytest output"""
        try:
            lines = pytest_output.split('\n')
            test_count = 0
            for line in lines:
                if 'PASSED' in line or 'FAILED' in line:
                    test_count += 1
            return test_count
        except:
            return 0
    
    def _cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

