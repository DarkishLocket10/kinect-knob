"""Capture backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import Frame


class CaptureError(RuntimeError):
    """Device missing / failed to open, with a user-actionable message."""


class CaptureBase(ABC):
    name = "base"
    has_depth = False

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Block until the next frame (or ~1s timeout -> None). rgb is RGB order,
        NOT mirrored (mirroring is applied centrally in the capture thread)."""

    @abstractmethod
    def stop(self) -> None: ...
