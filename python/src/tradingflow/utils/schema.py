"""Bidirectional name-to-position mapping for array axes."""

from __future__ import annotations

from collections.abc import Iterable


class Schema:
    """Bidirectional name-to-position mapping for a single array axis.

    Parameters
    ----------
    names
        Ordered iterable of unique string labels.

    Raises
    ------
    ValueError
        If any name appears more than once.
    """

    __slots__ = ("_names", "_lookup")

    def __init__(self, names: Iterable[str]) -> None:
        self._names: list[str] = list(names)
        self._lookup: dict[str, int] = {}
        for i, name in enumerate(self._names):
            if name in self._lookup:
                raise ValueError(f"duplicate name in schema: {name}")
            self._lookup[name] = i

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"Schema({self._names!r})"

    def index(self, name: str) -> int:
        """Look up the position of `name`.

        Raises
        ------
        KeyError
            If `name` is not in the schema.
        """
        return self._lookup[name]

    def indices(self, names: Iterable[str]) -> list[int]:
        """Resolve multiple names to positions.

        Raises
        ------
        KeyError
            If any name is not in the schema.
        """
        return [self._lookup[n] for n in names]

    def try_index(self, name: str) -> int | None:
        """Look up the position of `name`, returning `None` if absent."""
        return self._lookup.get(name)

    def name(self, index: int) -> str:
        """Look up the name at `index`.

        Raises
        ------
        IndexError
            If `index` is out of bounds.
        """
        return self._names[index]

    @property
    def names(self) -> list[str]:
        """All names in order."""
        return list(self._names)

    def contains(self, name: str) -> bool:
        """Whether the schema contains `name`."""
        return name in self._lookup

    def select(self, indices: Iterable[int]) -> Schema:
        """Create a sub-schema by selecting names at the given positions."""
        return Schema(self._names[i] for i in indices)

    def concat(self, other: Schema) -> Schema:
        """Create a schema by concatenating this schema with `other`.

        Raises
        ------
        ValueError
            If any name appears in both schemas.
        """
        return Schema([*self._names, *other._names])
