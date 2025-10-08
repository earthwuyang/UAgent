"""Java dependency parser using regex patterns."""

import logging
import re
from typing import List, Optional

from .base import BaseDependencyParser
from openhands.core.dependency_analyzer.models import ImportStatement


class JavaDependencyParser(BaseDependencyParser):
    """Java dependency parser using regex patterns."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize Java parser."""
        super().__init__(logger)

    def supports_language(self, language: str) -> bool:
        """Check if parser supports given language."""
        return language == 'java'

    async def parse(self, file_path: str, content: str) -> List[ImportStatement]:
        """Parse Java file content and extract imports using regex.
        
        Args:
            file_path: Path to the file being parsed
            content: Content of the Java file
            
        Returns:
            List of ImportStatement objects found in the file
        """
        imports = []
        
        try:
            imports = self._extract_imports_with_regex(content)
        except Exception as e:
            self.logger.error(f"Error parsing Java file {file_path}: {e}")
            return []
        
        # Filter out standard library and third-party imports
        filtered_imports = []
        for import_stmt in imports:
            if (not self.is_standard_library(import_stmt.module, 'java') and 
                not self.is_third_party(import_stmt.module, 'java')):
                filtered_imports.append(import_stmt)
            else:
                self.logger.debug(f"Skipping standard/third-party import: {import_stmt.module}")
        
        return filtered_imports

    def _extract_imports_with_regex(self, content: str) -> List[ImportStatement]:
        """Extract Java imports using regex patterns.
        
        Args:
            content: Java file content
            
        Returns:
            List of ImportStatement objects
        """
        imports = []
        
        # Regex pattern for Java import statements
        # Handles: import package.Class;, import static package.Class.method;, import package.*;
        import_pattern = r'^\s*import\s+(static\s+)?([\w.]+)(\*)?\s*;'
        
        for line_num, line in enumerate(content.splitlines(), 1):
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
            
            match = re.match(import_pattern, line)
            if match:
                is_static = match.group(1) is not None
                package_or_class = match.group(2)
                is_wildcard = match.group(3) is not None
                
                # Determine import type and confidence
                import_type = 'static'  # All Java imports are static at compile time
                confidence = 0.8 if is_wildcard else 1.0
                
                import_stmt = self._create_import_statement(
                    module=package_or_class,
                    import_type=import_type,
                    line_number=line_num,
                    confidence=confidence,
                    source='ast'  # Use 'ast' to maintain consistency, even though it's regex
                )
                
                # Add metadata
                if is_static:
                    import_stmt.metadata['static'] = True
                if is_wildcard:
                    import_stmt.metadata['wildcard'] = True
                    # For wildcard imports, we can't resolve to specific files easily
                    # So we mark them with lower confidence
                    import_stmt.confidence = 0.6
                
                imports.append(import_stmt)
        
        return imports
