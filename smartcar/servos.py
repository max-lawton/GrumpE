"""Servos.

    Actuator
    `-- Servo                  one hobby servo, angle 0..180, soft limits
        |-- ScanServo         front servo carrying the ultrasonic sensor
        `-- CameraServo       camera servo, nudged in small steps
            |-- CameraTilt    up / down   (limited to 35..180)
            `-- CameraPan     left / right (0..180)
    CameraGimbal              pan + tilt together
"""
from __future__ import annotations

import time

import gpiozero as gz

from .base import Actuator, clamp
from .pins import Pins


class Servo(Actuator):
    """Hobby servo using the stock pulse mapping: 0 deg = 0.5 ms and
    180 deg = 2.48 ms (500 + 11 us per degree) on a 20 ms frame.

    A servo only holds position while it is receiving pulses. detach() stops
    them, which silences buzzing and jitter; the servo then stays put (or
    sags under load).
    """

    CENTER = 90
    LIMIT_MIN = 0
    LIMIT_MAX = 180

    def __init__(self, pin: int, *, limits: tuple[int, int] | None = None,
                 name: str | None = None):
        super().__init__(name)
        self.pin = pin
        self.min_angle, self.max_angle = limits or (self.LIMIT_MIN, self.LIMIT_MAX)
        self._dev = self._register(
            gz.AngularServo(
                pin, initial_angle=None,          # no pulses until first move
                min_angle=0, max_angle=180,
                min_pulse_width=0.0005, max_pulse_width=0.00248,
            )
        )

    @property
    def angle(self) -> float | None:
        """Current target angle, or None if detached."""
        return self._dev.angle

    def move_to(self, angle: float, *, wait: float = 0.0, release: bool = False) -> float:
        """Move to `angle` (clamped to the soft limits). `wait` sleeps so the
        servo can arrive (~0.3 s for a big swing); `release` then detaches.
        Returns the angle actually commanded."""
        angle = clamp(angle, self.min_angle, self.max_angle)
        self._dev.angle = angle
        if wait:
            time.sleep(wait)
        if release:
            self.detach()
        return angle

    def step(self, delta: float, **kwargs) -> float:
        base = self.angle if self.angle is not None else self.CENTER
        return self.move_to(base + delta, **kwargs)

    def center(self, **kwargs) -> float:
        return self.move_to(self.CENTER, **kwargs)

    def detach(self) -> None:
        self._dev.detach()

    def on_close(self) -> None:
        self.detach()


class ScanServo(Servo):
    """Front servo that swings the ultrasonic sensor.
    In the stock code 0 deg = right and 180 deg = left."""

    RIGHT = 0
    FRONT = 90
    LEFT = 180

    def __init__(self, pin: int = Pins.SERVO_FRONT, **kwargs):
        kwargs.setdefault("name", "scan_servo")
        super().__init__(pin, **kwargs)

    def look_left(self, **kwargs) -> float:
        return self.move_to(self.LEFT, **kwargs)

    def look_right(self, **kwargs) -> float:
        return self.move_to(self.RIGHT, **kwargs)

    def look_front(self, **kwargs) -> float:
        return self.move_to(self.FRONT, **kwargs)


class CameraServo(Servo):
    """Camera servo moved in small increments, like the Bluetooth app does."""

    DEFAULT_STEP = 1

    def increase(self, step: float | None = None) -> float:
        return self.step(self.DEFAULT_STEP if step is None else abs(step))

    def decrease(self, step: float | None = None) -> float:
        return self.step(-(self.DEFAULT_STEP if step is None else abs(step)))


class CameraTilt(CameraServo):
    """Camera up/down. Stock code never lets it below 35 deg."""

    LIMIT_MIN = 35
    LIMIT_MAX = 180

    def __init__(self, pin: int = Pins.CAM_TILT, **kwargs):
        kwargs.setdefault("name", "cam_tilt")
        super().__init__(pin, **kwargs)

    def up(self, step: float | None = None) -> float:
        return self.increase(step)

    def down(self, step: float | None = None) -> float:
        return self.decrease(step)


class CameraPan(CameraServo):
    """Camera left/right. Larger angle = further left (stock servo_left())."""

    LIMIT_MIN = 0
    LIMIT_MAX = 180

    def __init__(self, pin: int = Pins.CAM_PAN, **kwargs):
        kwargs.setdefault("name", "cam_pan")
        super().__init__(pin, **kwargs)

    def left(self, step: float | None = None) -> float:
        return self.increase(step)

    def right(self, step: float | None = None) -> float:
        return self.decrease(step)


class CameraGimbal(Actuator):
    """Pan + tilt as one unit."""

    def __init__(self, pan: CameraPan | None = None, tilt: CameraTilt | None = None,
                 *, name: str | None = None):
        super().__init__(name or "camera")
        self.pan = self._adopt(pan or CameraPan())
        self.tilt = self._adopt(tilt or CameraTilt())

    def look(self, pan: float | None = None, tilt: float | None = None, **kwargs) -> None:
        if pan is not None:
            self.pan.move_to(pan, **kwargs)
        if tilt is not None:
            self.tilt.move_to(tilt, **kwargs)

    def center(self, **kwargs) -> None:
        self.pan.center(**kwargs)
        self.tilt.center(**kwargs)

    def detach(self) -> None:
        self.pan.detach()
        self.tilt.detach()
