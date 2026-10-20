"""Bluetooth serial link (the Yahboom phone app protocol: packets are
framed as '$ ... #')."""
from __future__ import annotations

from .base import Component
from .pins import Pins


class BluetoothLink(Component):
    START = "$"
    END = "#"
    MAX_LEN = 80            # the stock parser drops anything longer

    def __init__(self, port: str = Pins.SERIAL_PORT, baud: int = Pins.SERIAL_BAUD,
                 *, ser=None, name: str | None = None):
        """`ser` lets you inject any pyserial-like object (used by the tests)."""
        super().__init__(name or "bluetooth")
        if ser is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("Bluetooth needs pyserial: pip install pyserial") from exc
            ser = serial.Serial(port, baud, timeout=0)
        self._ser = ser
        self._buf = ""
        self._started = False

    def poll(self) -> list[str]:
        """Read whatever has arrived and return the complete packets
        (each including its '$' and '#'). Partial packets are kept for the
        next call."""
        packets: list[str] = []
        waiting = self._ser.in_waiting
        if not waiting:
            return packets
        for ch in self._ser.read(waiting).decode("latin-1"):
            if ch == self.START:
                self._buf, self._started = ch, True
                continue
            if not self._started:
                continue
            self._buf += ch
            if ch == self.END:
                packets.append(self._buf)
                self._buf, self._started = "", False
            elif len(self._buf) >= self.MAX_LEN:
                self._buf, self._started = "", False
        return packets

    def send(self, text: str) -> None:
        self._ser.write(text.encode("latin-1"))

    def on_close(self) -> None:
        self._ser.close()
