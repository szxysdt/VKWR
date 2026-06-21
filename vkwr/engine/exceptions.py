class EngineGenerateError(Exception):
    """Raised when AsyncLLMEngine.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        super().__init__(
            "EngineCore encountered an issue. See stack trace (above) for the root cause.",
            *args,
            **kwargs,
        )
        self.__suppress_context__ = suppress_context
