__all__ = ["GeometryCollectionP", "RodCollection", "create_rod_collection"]

from typing import TYPE_CHECKING, Protocol
from typing_extensions import Self

import numpy as np

# TODO
# from .rod import Rod


# TODO
class GeometryCollectionP(Protocol):
    @property
    def tag(self) -> str: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> "Geometry": ...

    def __iter__(self): ...

    @classmethod
    def create_collection(cls, number: int, tag: str = None) -> Self:
        """
        Create a collection of geometries
        """
        ...


# TODO
class RodCollection:
    def __init__(self):
        self._tag: str = ""
        self._rods: list["Rod"] = []

    @property
    def tag(self) -> str:
        return self._tag

    def __len__(self) -> int:
        return len(self._rods)

    def __getitem__(self, index: int) -> "Rod":
        return self._rods[index]

    def __iter__(self):
        return iter(self._rods)

    @classmethod
    def create_collection(cls, number: int, tag: str = None) -> Self:
        pass

    def update_history(
        self, keytimes: np.ndarray, position: np.ndarray, radius: np.ndarray
    ):
        pass


# Alias for factory functions
create_rod_collection = RodCollection.create_collection


if TYPE_CHECKING:
    _: GeometryCollectionP = RodCollection(...)
