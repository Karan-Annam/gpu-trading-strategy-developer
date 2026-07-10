class CompileError(Exception):
    """Compilation failure with a source location (1-based) for editor squiggles."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def __str__(self) -> str:
        if self.line:
            return f"line {self.line}:{self.col}: {self.message}"
        return self.message
