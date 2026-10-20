"""Sensors.

    Sensor
    |-- DigitalSensor           one digital input, polarity set per subclass
    |   |-- LineSensor          active LOW  (black line under it)
    |   |-- ObstacleSensor      active LOW  (obstacle in front)
    |   `-- LightSensor         active HIGH (light seen)
    |-- SensorArray             ordered group of sensors
    |   |-- LineSensorArray     the 4-channel tracker
    |   `-- SidedSensorPair     left/right pair
    |       |-- ObstacleSensorPair
    |       `-- LightSensorPair
    `-- UltrasonicSensor        distance in cm
"""
from __future__ import annotations

import time

import gpiozero as gz

from .base import Sensor
from .pins import Pins


# --------------------------------------------------------------------------
# Single digital sensors
# --------------------------------------------------------------------------

class DigitalSensor(Sensor):
    """One digital input. `active` is True when the sensor is triggered."""

    ACTIVE_HIGH = True

    def __init__(self, pin: int, *, name: str | None = None):
        super().__init__(name)
        self.pin = pin
        # The modules drive the line themselves, so the internal pull only
        # sets the idle level. It must match the polarity: BCM 2 and 3 have
        # hard-wired pull-ups (the tracker's outer-left sensor is on BCM 3),
        # and gpiozero raises PinFixedPull if you ask those for anything else.
        # Active-low sensors -> pull-up; active-high sensors -> pull-down.
        self._dev = self._register(
            gz.DigitalInputDevice(pin, pull_up=not self.ACTIVE_HIGH)
        )

    @property
    def active(self) -> bool:
        return bool(self._dev.value)

    def read(self) -> bool:
        return self.active


class LineSensor(DigitalSensor):
    """IR reflectance sensor. LOW (indicator lit) = black line underneath."""
    ACTIVE_HIGH = False

    @property
    def on_line(self) -> bool:
        return self.active


class ObstacleSensor(DigitalSensor):
    """IR proximity sensor. LOW (indicator lit) = obstacle ahead."""
    ACTIVE_HIGH = False

    @property
    def blocked(self) -> bool:
        return self.active


class LightSensor(DigitalSensor):
    """Digital LDR module. HIGH = light seen above the module's threshold."""
    ACTIVE_HIGH = True

    @property
    def lit(self) -> bool:
        return self.active


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------

class SensorArray(Sensor):
    """Ordered group of sensors, read together.

    bits() packs the readings into an int with the FIRST sensor as the most
    significant bit, so a line tracker reading outer_left..outer_right as
    0110 gives 0b0110, the same convention as the TRACK_* masks in the C code.
    """

    def __init__(self, sensors: dict, name: str | None = None):
        super().__init__(name)
        self.sensors = {key: self._adopt(s) for key, s in sensors.items()}

    def __getitem__(self, key: str):
        return self.sensors[key]

    def __iter__(self):
        return iter(self.sensors.values())

    def read(self) -> dict:
        return {key: s.read() for key, s in self.sensors.items()}

    def bits(self) -> int:
        value = 0
        for s in self.sensors.values():
            value = (value << 1) | int(bool(s.read()))
        return value

    def pattern(self) -> str:
        return "".join("1" if s.read() else "0" for s in self.sensors.values())

    def any_active(self) -> bool:
        return any(s.read() for s in self.sensors.values())


class LineSensorArray(SensorArray):
    NAMES = ("outer_left", "inner_left", "inner_right", "outer_right")

    def __init__(self, pins=Pins.TRACK, name: str | None = None):
        sensors = {n: LineSensor(p, name=n) for n, p in zip(self.NAMES, pins)}
        super().__init__(sensors, name or "line_tracker")

    @property
    def outer_left(self) -> bool:
        return self.sensors["outer_left"].on_line

    @property
    def inner_left(self) -> bool:
        return self.sensors["inner_left"].on_line

    @property
    def inner_right(self) -> bool:
        return self.sensors["inner_right"].on_line

    @property
    def outer_right(self) -> bool:
        return self.sensors["outer_right"].on_line


class SidedSensorPair(SensorArray):
    """Left/right pair of identical digital sensors."""

    SENSOR_CLASS = DigitalSensor
    DEFAULT_PINS: tuple = ()

    def __init__(self, left_pin: int | None = None, right_pin: int | None = None,
                 name: str | None = None):
        left_default, right_default = self.DEFAULT_PINS
        cls = self.SENSOR_CLASS
        super().__init__(
            {
                "left": cls(left_default if left_pin is None else left_pin, name="left"),
                "right": cls(right_default if right_pin is None else right_pin, name="right"),
            },
            name,
        )

    @property
    def left(self):
        return self.sensors["left"]

    @property
    def right(self):
        return self.sensors["right"]


class ObstacleSensorPair(SidedSensorPair):
    SENSOR_CLASS = ObstacleSensor
    DEFAULT_PINS = (Pins.AVOID_LEFT, Pins.AVOID_RIGHT)


class LightSensorPair(SidedSensorPair):
    SENSOR_CLASS = LightSensor
    DEFAULT_PINS = (Pins.LDR_LEFT, Pins.LDR_RIGHT)


# --------------------------------------------------------------------------
# Ultrasonic
# --------------------------------------------------------------------------

class UltrasonicSensor(Sensor):
    """HC-SR04 style ranger. Distances are in centimetres; None = no echo."""

    def __init__(self, trigger: int = Pins.ULTRA_TRIG, echo: int = Pins.ULTRA_ECHO,
                 *, max_distance_cm: float = 500, name: str | None = None):
        super().__init__(name or "ultrasonic")
        self.max_distance_cm = max_distance_cm
        self._dev = self._register(
            gz.DistanceSensor(
                echo=echo, trigger=trigger,
                max_distance=max_distance_cm / 100.0,
                queue_len=1,                      # raw, unsmoothed reading
            )
        )

    def read(self) -> float | None:
        """One raw reading, or None if nothing was in range."""
        cm = self._dev.distance * 100.0
        return None if cm >= self.max_distance_cm - 0.5 else cm

    @property
    def distance_cm(self) -> float | None:
        return self.read()

    def read_filtered(self, samples: int = 5, delay: float = 0.06) -> float | None:
        """Median-style filter from the stock code: take `samples` valid
        readings, drop the smallest and largest, average the rest.
        Returns None if too few valid readings arrive."""
        values: list[float] = []
        for _ in range(samples * 5):              # bounded, never loops forever
            cm = self.read()
            if cm is not None and cm > 0:
                values.append(cm)
                if len(values) == samples:
                    break
            time.sleep(delay)
        if len(values) < samples:
            return None
        values.sort()
        if samples >= 3:
            values = values[1:-1]
        return sum(values) / len(values)
