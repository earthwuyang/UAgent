"""Custom exceptions for dependency analysis."""


class DependencyAnalysisError(Exception):
    """Base exception for dependency analysis errors."""
    pass


class ParserError(DependencyAnalysisError):
    """Raised when parsing fails."""
    
    def __init__(self, file_path: str, language: str, message: str):
        self.file_path = file_path
        self.language = language
        super().__init__(f"Parser error in {file_path} ({language}): {message}")


class ImportResolutionError(DependencyAnalysisError):
    """Raised when import path cannot be resolved."""
    
    def __init__(self, import_str: str, source_file: str, message: str):
        self.import_str = import_str
        self.source_file = source_file
        super().__init__(f"Cannot resolve import '{import_str}' in {source_file}: {message}")


class CacheError(DependencyAnalysisError):
    """Raised when cache operation fails."""
    pass


class GraphBuildError(DependencyAnalysisError):
    """Raised when graph construction fails."""
    pass


class UnsupportedLanguageError(DependencyAnalysisError):
    """Raised when language is not supported."""
    
    def __init__(self, language: str):
        self.language = language
        super().__init__(f"Unsupported language: {language}")
