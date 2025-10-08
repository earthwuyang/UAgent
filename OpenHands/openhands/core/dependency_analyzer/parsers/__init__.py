"""Package initialization for language parsers."""

from typing import Optional, Dict, Type

from .base import BaseDependencyParser
from .python_parser import PythonDependencyParser
from .javascript_parser import JavaScriptDependencyParser, TypeScriptDependencyParser
from .java_parser import JavaDependencyParser

# Export all parser classes
__all__ = [
    'BaseDependencyParser',
    'PythonDependencyParser',
    'JavaScriptDependencyParser', 
    'TypeScriptDependencyParser',
    'JavaDependencyParser',
    'get_parser',
    'PARSER_REGISTRY'
]

# Parser registry mapping file extensions to parser classes
PARSER_REGISTRY: Dict[str, Type[BaseDependencyParser]] = {
    '.py': PythonDependencyParser,
    '.js': JavaScriptDependencyParser,
    '.jsx': JavaScriptDependencyParser,
    '.ts': TypeScriptDependencyParser,
    '.tsx': TypeScriptDependencyParser,
    '.java': JavaDependencyParser,
}

# Language to parser mapping
LANGUAGE_PARSER_REGISTRY: Dict[str, Type[BaseDependencyParser]] = {
    'python': PythonDependencyParser,
    'javascript': JavaScriptDependencyParser,
    'typescript': TypeScriptDependencyParser,
    'java': JavaDependencyParser,
}


def get_parser(file_path: str) -> Optional[BaseDependencyParser]:
    """Get parser based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Parser instance or None if no parser available for file type
    """
    # Extract file extension
    if '.' not in file_path:
        return None
    
    ext = '.' + file_path.split('.')[-1].lower()
    
    # Get parser class from registry
    parser_class = PARSER_REGISTRY.get(ext)
    if parser_class:
        return parser_class()
    
    return None


def get_parser_by_language(language: str) -> Optional[BaseDependencyParser]:
    """Get parser based on language name.
    
    Args:
        language: Language name (python, javascript, typescript, java)
        
    Returns:
        Parser instance or None if no parser available for language
    """
    parser_class = LANGUAGE_PARSER_REGISTRY.get(language.lower())
    if parser_class:
        return parser_class()
    
    return None


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language name or None if language cannot be detected
    """
    if '.' not in file_path:
        return None
    
    ext = '.' + file_path.split('.')[-1].lower()
    
    # Extension to language mapping
    ext_to_language = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
    }
    
    return ext_to_language.get(ext)
