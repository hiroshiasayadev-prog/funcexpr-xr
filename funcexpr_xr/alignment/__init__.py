from .base import AlignmentStrategy
from .exact import exact_align
from .inner import inner_align
from .outer import outer_align

ALIGNMENT_STRATEGIES: dict[str, AlignmentStrategy] = {
    "exact": exact_align,
    "inner": inner_align,
    "outer": outer_align,
}


def register(name: str, strategy: AlignmentStrategy) -> None:
    """Register a custom alignment strategy.

    Args:
        name: The strategy name used in ``fxr.evaluate(alignment=name)``.
        strategy: A callable conforming to :class:`AlignmentStrategy`.

    Example:
        >>> from fxr.alignment import register
        >>> def my_interp(arrays):
        ...     ...
        >>> register("interp", my_interp)
    """
    ALIGNMENT_STRATEGIES[name] = strategy


__all__ = [
    "AlignmentStrategy",
    "ALIGNMENT_STRATEGIES",
    "register",
    "exact_align",
    "inner_align",
    "outer_align",
]