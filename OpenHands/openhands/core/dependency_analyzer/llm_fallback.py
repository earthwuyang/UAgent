"""LLM-based dependency extraction for complex cases using existing LLM infrastructure."""

import asyncio
import json
import logging
import time
from typing import List, Optional, Dict, Any

try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

from openhands.llm.llm import LLM
from openhands.core.config.llm_config import LLMConfig

from .models import ImportStatement
from .exceptions import DependencyAnalysisError


class LLMDependencyExtractor:
    """Uses LLM to extract dependencies from code when AST parsing fails."""

    def __init__(
        self,
        llm_config: LLMConfig,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize LLM dependency extractor.
        
        Args:
            llm_config: LLM configuration
            logger: Optional logger
        """
        self.llm_config = llm_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize LLM instance with service_id
        self.llm = LLM(llm_config, service_id='dependency-analyzer')
        
        # Cache for LLM responses (content hash -> response)
        self._response_cache: Dict[str, List[ImportStatement]] = {}
        
        # Usage tracking
        self._usage_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_tokens_used': 0
        }

    async def extract_dependencies(
        self,
        file_path: str,
        content: str,
        language: str,
        context: Optional[str] = None
    ) -> List[ImportStatement]:
        """Use LLM to extract dependencies from code.
        
        Args:
            file_path: Path to the file being analyzed
            content: Content of the file
            language: Programming language
            context: Optional context about why LLM fallback is needed
            
        Returns:
            List of ImportStatement objects found by LLM
        """
        self._usage_stats['total_calls'] += 1
        
        # Check cache first
        content_hash = self._hash_content(content)
        cache_key = f"{language}:{content_hash}"
        
        if cache_key in self._response_cache:
            self._usage_stats['cache_hits'] += 1
            self.logger.debug(f"Using cached LLM response for {file_path}")
            return self._response_cache[cache_key]
        
        try:
            # Skip very large files to avoid excessive costs
            if len(content) > 50000:  # 50KB limit
                self.logger.warning(f"File {file_path} too large for LLM analysis ({len(content)} chars)")
                return []
            
            self.logger.info(f"Using LLM fallback for dependency extraction: {file_path}")
            
            # Generate prompt based on language
            prompt = self._create_prompt(content, language, file_path, context)
            
            # Make LLM call with timeout
            start_time = time.time()
            response = await asyncio.wait_for(
                self._call_llm(prompt),
                timeout=30.0  # 30 second timeout
            )
            
            analysis_time = time.time() - start_time
            
            # Parse response
            imports = self._parse_llm_response(response, language)
            
            # Apply confidence penalty for LLM-based extraction
            for import_stmt in imports:
                import_stmt.confidence *= 0.8  # Reduce confidence by 20%
                import_stmt.source = 'llm'
            
            # Cache response
            self._response_cache[cache_key] = imports
            
            # Update stats
            self._usage_stats['successful_extractions'] += 1
            
            self.logger.info(
                f"LLM extracted {len(imports)} imports from {file_path} "
                f"in {analysis_time:.2f}s"
            )
            
            return imports
            
        except asyncio.TimeoutError:
            self.logger.warning(f"LLM call timed out for {file_path}")
            self._usage_stats['failed_extractions'] += 1
            return []
        
        except Exception as e:
            self.logger.warning(f"LLM extraction failed for {file_path}: {e}")
            self._usage_stats['failed_extractions'] += 1
            return []

    def _create_prompt(
        self,
        content: str,
        language: str,
        file_path: str,
        context: Optional[str] = None
    ) -> str:
        """Create language-specific prompt for LLM.
        
        Args:
            content: File content
            language: Programming language
            file_path: File path for context
            context: Additional context
            
        Returns:
            Formatted prompt string
        """
        if language == 'python':
            return self._create_python_prompt(content, file_path, context)
        elif language in ('javascript', 'typescript'):
            return self._create_javascript_prompt(content, file_path, context, language)
        elif language == 'java':
            return self._create_java_prompt(content, file_path, context)
        else:
            return self._create_generic_prompt(content, file_path, language, context)

    def _create_python_prompt(self, content: str, file_path: str, context: Optional[str] = None) -> str:
        """Create Python-specific prompt."""
        context_info = f"\n\nContext: {context}" if context else ""
        
        return f"""You are a code analysis expert. Analyze the following Python code and extract ALL import statements.

For each import, provide:
- module: the module name (e.g., 'os', './utils', 'mypackage.submodule')
- import_type: 'static', 'dynamic', or 'conditional'  
- line_number: approximate line number
- confidence: your confidence (0.0-1.0) that this is a real import

Include:
- Standard imports: import os, from x import y
- Relative imports: from . import utils, from .. import config
- Dynamic imports: importlib.import_module(), __import__()
- Conditional imports: imports inside if/try blocks

Exclude:
- Comments
- String literals that look like imports but aren't
- Standard library modules (os, sys, json, etc.)
- Third-party packages (requests, pandas, numpy, etc.)

Return ONLY valid JSON array of objects with keys: module, import_type, line_number, confidence

File: {file_path}{context_info}

Code:
```python
{content}
```

JSON output:"""

    def _create_javascript_prompt(
        self, 
        content: str, 
        file_path: str, 
        context: Optional[str] = None,
        language: str = 'javascript'
    ) -> str:
        """Create JavaScript/TypeScript-specific prompt."""
        context_info = f"\n\nContext: {context}" if context else ""
        lang_name = "TypeScript" if language == 'typescript' else "JavaScript"
        
        return f"""You are a code analysis expert. Analyze the following {lang_name} code and extract ALL import statements.

For each import, provide:
- module: the module name (e.g., './utils', '../config', './components/Button')
- import_type: 'static', 'dynamic', or 'conditional'
- line_number: approximate line number  
- confidence: your confidence (0.0-1.0) that this is a real import

Include:
- ES6 imports: import {{ foo }} from './module'
- CommonJS requires: require('./module')  
- Dynamic imports: import('./module')
- Relative imports: ./file, ../directory/file

Exclude:
- Comments
- String literals that look like imports but aren't
- Node.js built-in modules (fs, path, http, etc.)
- NPM packages (react, lodash, express, etc.)

Return ONLY valid JSON array of objects with keys: module, import_type, line_number, confidence

File: {file_path}{context_info}

Code:
```{language}
{content}
```

JSON output:"""

    def _create_java_prompt(self, content: str, file_path: str, context: Optional[str] = None) -> str:
        """Create Java-specific prompt."""
        context_info = f"\n\nContext: {context}" if context else ""
        
        return f"""You are a code analysis expert. Analyze the following Java code and extract ALL import statements.

For each import, provide:
- module: the full class/package name (e.g., 'com.example.MyClass', 'com.example.utils.Helper')
- import_type: 'static' (all Java imports are static at compile time)
- line_number: approximate line number
- confidence: your confidence (0.0-1.0) that this is a real import

Include:
- Regular imports: import com.example.MyClass;
- Static imports: import static com.example.Utils.helper;
- Wildcard imports: import com.example.*;

Exclude:
- Comments  
- String literals that look like imports but aren't
- Java standard library (java.*, javax.*, org.w3c.*, etc.)
- Common third-party libraries (org.apache.*, org.springframework.*, etc.)

Return ONLY valid JSON array of objects with keys: module, import_type, line_number, confidence

File: {file_path}{context_info}

Code:
```java
{content}
```

JSON output:"""

    def _create_generic_prompt(
        self, 
        content: str, 
        file_path: str, 
        language: str, 
        context: Optional[str] = None
    ) -> str:
        """Create generic prompt for unsupported languages."""
        context_info = f"\n\nContext: {context}" if context else ""
        
        return f"""You are a code analysis expert. Analyze the following {language} code and extract import/dependency statements.

Look for any statements that import, include, require, or reference other files/modules.

For each dependency, provide:
- module: the module/file name being imported
- import_type: 'static', 'dynamic', or 'conditional'
- line_number: approximate line number  
- confidence: your confidence (0.0-1.0) that this is a real dependency

Exclude:
- Comments
- String literals
- Standard library modules
- Third-party packages

Return ONLY valid JSON array of objects with keys: module, import_type, line_number, confidence

File: {file_path}{context_info}

Code:
```{language}
{content}
```

JSON output:"""

    async def _call_llm(self, prompt: str) -> str:
        """Make LLM call with the provided prompt.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            LLM response string
        """
        try:
            # Create messages for chat completion
            messages = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Make LLM call using asyncio.to_thread since completion is sync
            response = await asyncio.to_thread(
                self.llm.completion,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2000  # Reasonable limit for import lists
            )
            
            # Track token usage if available
            if hasattr(response, 'usage') and response.usage:
                self._usage_stats['total_tokens_used'] += response.usage.total_tokens
            
            # Extract content from response
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            else:
                return str(response)
                
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise

    def _parse_llm_response(self, response: str, language: str) -> List[ImportStatement]:
        """Parse LLM response into ImportStatement objects.
        
        Args:
            response: Raw LLM response
            language: Programming language for context
            
        Returns:
            List of ImportStatement objects
        """
        imports = []
        
        try:
            # Try to extract JSON from response
            json_str = self._extract_json_from_response(response)
            
            if not json_str:
                self.logger.warning("No JSON found in LLM response")
                return self._regex_fallback_parse(response, language)
            
            # Parse JSON - try json_repair first if available
            import_data = None
            try:
                import_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.debug(f"Direct JSON parsing failed: {e}")
                if JSON_REPAIR_AVAILABLE:
                    try:
                        repaired_json = json_repair.repair_json(json_str)
                        import_data = json.loads(repaired_json)
                        self.logger.debug("Successfully repaired malformed JSON")
                    except Exception as repair_error:
                        self.logger.debug(f"JSON repair failed: {repair_error}")
                
                if import_data is None:
                    self.logger.warning("Failed to parse LLM JSON response, falling back to regex")
                    return self._regex_fallback_parse(response, language)
            
            if not isinstance(import_data, list):
                self.logger.warning("LLM response is not a JSON array")
                return self._regex_fallback_parse(response, language)
            
            # Convert to ImportStatement objects
            for item in import_data:
                if not isinstance(item, dict):
                    continue
                
                # Validate required fields
                if 'module' not in item or 'import_type' not in item:
                    continue
                
                try:
                    import_stmt = ImportStatement(
                        module=str(item['module']),
                        import_type=str(item.get('import_type', 'static')),
                        line_number=int(item.get('line_number', 1)),
                        confidence=float(item.get('confidence', 0.7)),
                        source='llm'
                    )
                    imports.append(import_stmt)
                    
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Skipping invalid import from LLM: {item} - {e}")
                    continue
            
            return imports
            
        except Exception as e:
            self.logger.warning(f"Error parsing LLM response: {e}")
            return self._regex_fallback_parse(response, language)

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON array from LLM response text.
        
        Args:
            response: Raw LLM response
            
        Returns:
            JSON string or None if not found
        """
        # Look for JSON array patterns
        import re
        
        # Try to find JSON array (starts with [ and ends with ])
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            # Return the longest match (most complete)
            return max(matches, key=len)
        
        # Try to find just the array content between brackets
        bracket_pattern = r'\[(.*?)\]'
        bracket_matches = re.findall(bracket_pattern, response, re.DOTALL)
        
        if bracket_matches:
            return f"[{bracket_matches[0]}]"
        
        return None

    def _regex_fallback_parse(self, response: str, language: str) -> List[ImportStatement]:
        """Fallback regex parsing of LLM response.
        
        Args:
            response: Raw LLM response text
            language: Programming language
            
        Returns:
            List of ImportStatement objects
        """
        imports = []
        
        try:
            # Look for patterns that might be imports in the text
            import re
            
            # Common patterns that might indicate imports
            patterns = [
                r"module['\"]?\s*:\s*['\"]([^'\"]+)['\"]",  # module: "name"
                r"import[^'\"]*['\"]([^'\"]+)['\"]",         # import "name"
                r"from[^'\"]*['\"]([^'\"]+)['\"]",          # from "name"
                r"require[^'\"]*['\"]([^'\"]+)['\"]",       # require "name"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    if match and not match.startswith(('http', 'https', 'ftp')):
                        import_stmt = ImportStatement(
                            module=match,
                            import_type='static',
                            line_number=1,
                            confidence=0.5,  # Low confidence for regex extraction
                            source='llm'
                        )
                        imports.append(import_stmt)
            
            return imports[:10]  # Limit to 10 imports max
            
        except Exception as e:
            self.logger.debug(f"Regex fallback parsing failed: {e}")
            return []

    def _hash_content(self, content: str) -> str:
        """Create hash of content for caching.
        
        Args:
            content: File content
            
        Returns:
            Content hash string
        """
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        total_calls = self._usage_stats['total_calls']
        cache_hit_rate = (
            self._usage_stats['cache_hits'] / total_calls 
            if total_calls > 0 else 0
        )
        success_rate = (
            self._usage_stats['successful_extractions'] / total_calls
            if total_calls > 0 else 0
        )
        
        return {
            'total_calls': total_calls,
            'cache_hits': self._usage_stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'successful_extractions': self._usage_stats['successful_extractions'],
            'failed_extractions': self._usage_stats['failed_extractions'],
            'success_rate': success_rate,
            'total_tokens_used': self._usage_stats['total_tokens_used'],
            'cached_responses': len(self._response_cache)
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._response_cache.clear()
        self.logger.debug("Cleared LLM response cache")
