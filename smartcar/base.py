"""Base classes shared by every piece of car hardware."""
from __future__ import annotations

from abc import ABC, abstractmethod


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class Component(ABC):
    """Anything that owns hardware.

    Subclasses register the gpiozero devices they create with _register() and
    any sub-components with _adopt(). close() releases everything in reverse
    order, so `with Car() as car:` always leaves the pins free (and the motors
    stopped, via on_close()).
    """

    def __init__(self, name: str | None = None):
        self.name = name or type(self).__name__
        self._devices: list = []
        self._children: list[Component] = []
        self._closed = False

    def _register(self, device):
        self._devices.append(device)
        return device

    def _adopt(self, child: "Component"):
        self._children.append(child)
        return child

    def on_close(self) -> None:
        """Hook: put hardware in a safe state before the pins are released."""

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.on_close()
        for child in reversed(self._children):
            child.close()
        for device in reversed(self._devices):
            device.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"


class Sensor(Component):
    """A component that is read."""

    @abstractmethod
    def read(self):
        ...


class Actuator(Component):
    """A component that is commanded."""
