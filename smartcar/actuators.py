"""Non-servo actuators.

    Actuator
    |-- Motor        one H-bridge channel, speed -1..1
    |-- DriveTrain   left + right Motor, with the stock maneuvers
    |-- RGBLed       0..255 per channel
    |-- Fan          fire-extinguisher fan (active LOW)
    `-- KeyBuzzer    onboard button + buzzer, which share one GPIO
"""
from __future__ import annotations

import time

import gpiozero as gz

from .base import Actuator, clamp
from .pins import Pins


class Motor(Actuator):
    """One motor channel: two digital direction pins plus a PWM speed pin."""

    def __init__(self, forward_pin: int, backward_pin: int, pwm_pin: int,
                 *, pwm_frequency: int = 100, name: str | None = None):
        super().__init__(name)
        self._fwd = self._register(gz.DigitalOutputDevice(forward_pin))
        self._back = self._register(gz.DigitalOutputDevice(backward_pin))
        self._pwm = self._register(gz.PWMOutputDevice(pwm_pin, frequency=pwm_frequency))
        self._speed = 0.0

    @property
    def speed(self) -> float:
        return self._speed

    def set(self, speed: float) -> None:
        """speed: -1.0 (full reverse) .. 0 (coast) .. 1.0 (full forward)."""
        speed = clamp(float(speed), -1.0, 1.0)
        if speed > 0:
            self._fwd.on()
            self._back.off()
        elif speed < 0:
            self._fwd.off()
            self._back.on()
        else:
            self._fwd.off()
            self._back.off()
        self._pwm.value = abs(speed)
        self._speed = speed

    def forward(self, speed: float = 1.0) -> None:
        self.set(abs(speed))

    def backward(self, speed: float = 1.0) -> None:
        self.set(-abs(speed))

    def stop(self) -> None:
        self.set(0.0)

    def on_close(self) -> None:
        self.stop()


class DriveTrain(Actuator):
    """Differential drive. Speeds are -1..1; the stock 150/255 is about 0.6.

    Maneuvers match the stock C functions:
        forward / backward   both sides same direction
        arc_left / arc_right one side stopped (stock left()/right())
        spin_left/spin_right sides opposite (turn in place)
    """

    def __init__(self, left: Motor | None = None, right: Motor | None = None,
                 *, default_speed: float = 0.6, name: str | None = None):
        super().__init__(name or "drive")
        self.default_speed = default_speed
        self.left = self._adopt(left or Motor(
            Pins.LEFT_MOTOR_GO, Pins.LEFT_MOTOR_BACK, Pins.LEFT_MOTOR_PWM, name="left_motor"))
        self.right = self._adopt(right or Motor(
            Pins.RIGHT_MOTOR_GO, Pins.RIGHT_MOTOR_BACK, Pins.RIGHT_MOTOR_PWM, name="right_motor"))

    def _s(self, speed):
        return self.default_speed if speed is None else abs(speed)

    def drive(self, left: float, right: float) -> None:
        self.left.set(left)
        self.right.set(right)

    def steer(self, speed: float, turn: float) -> None:
        """Arcade-style mix: turn > 0 curves right, turn < 0 curves left."""
        self.drive(clamp(speed + turn, -1, 1), clamp(speed - turn, -1, 1))

    def forward(self, speed: float | None = None) -> None:
        s = self._s(speed)
        self.drive(s, s)

    def backward(self, speed: float | None = None) -> None:
        s = self._s(speed)
        self.drive(-s, -s)

    def arc_left(self, speed: float | None = None) -> None:
        self.drive(0, self._s(speed))

    def arc_right(self, speed: float | None = None) -> None:
        self.drive(self._s(speed), 0)

    def spin_left(self, speed: float | None = None) -> None:
        s = self._s(speed)
        self.drive(-s, s)

    def spin_right(self, speed: float | None = None) -> None:
        s = self._s(speed)
        self.drive(s, -s)

    def stop(self) -> None:
        self.drive(0, 0)

    def on_close(self) -> None:
        self.stop()


class RGBLed(Actuator):
    """Common-cathode RGB LED driven with PWM. Channels are 0..255."""

    COLORS = {
        "off": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0),
        "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255),
        "magenta": (255, 0, 255), "white": (255, 255, 255),
    }

    def __init__(self, red: int = Pins.LED_R, green: int = Pins.LED_G,
                 blue: int = Pins.LED_B, *, name: str | None = None):
        super().__init__(name or "led")
        self._dev = self._register(gz.RGBLED(red, green, blue, pwm=True))

    def set(self, r: int, g: int, b: int) -> None:
        self._dev.color = tuple(clamp(c, 0, 255) / 255.0 for c in (r, g, b))

    def set_color(self, color: str) -> None:
        self.set(*self.COLORS[color])

    @property
    def rgb(self) -> tuple[int, int, int]:
        r, g, b = self._dev.value
        return round(r * 255), round(g * 255), round(b * 255)

    def off(self) -> None:
        self.set(0, 0, 0)

    def on_close(self) -> None:
        self.off()


class Fan(Actuator):
    """Fire-extinguisher fan. The stock code idles the pin HIGH, so the
    output is active LOW."""

    def __init__(self, pin: int = Pins.OUTFIRE, *, name: str | None = None):
        super().__init__(name or "fan")
        self._dev = self._register(
            gz.DigitalOutputDevice(pin, active_high=False, initial_value=False))

    @property
    def is_on(self) -> bool:
        return bool(self._dev.value)

    def on(self) -> None:
        self._dev.on()

    def off(self) -> None:
        self._dev.off()

    def toggle(self) -> None:
        self._dev.toggle()

    def on_close(self) -> None:
        self.off()


class KeyBuzzer(Actuator):
    """The onboard button and buzzer share one GPIO (wiringPi 10).

    gpiozero won't let two devices own the same pin, so this one component
    owns it and flips the pin between input (button, pulled up, LOW =
    pressed) and output (buzzer, LOW = sound) as needed. The pin is always
    returned to input so the button keeps working.
    """

    def __init__(self, pin: int = Pins.KEY_BUZZER, *, debounce: float = 0.02,
                 name: str | None = None):
        super().__init__(name or "key_buzzer")
        self.debounce = debounce
        ensure = getattr(gz.Device, "ensure_pin_factory", None)
        if ensure:
            ensure()
        self._pin = gz.Device.pin_factory.pin(pin)
        self._to_input()

    def _to_input(self) -> None:
        self._pin.function = "input"
        self._pin.pull = "up"

    # ---- button
    @property
    def is_pressed(self) -> bool:
        return not self._pin.state

    def wait_for_press(self, timeout: float | None = None) -> bool:
        """Block until a debounced press and release. False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.is_pressed:
                time.sleep(self.debounce)
                if self.is_pressed:
                    break
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        while self.is_pressed:                    # wait for release
            time.sleep(0.005)
        time.sleep(0.05)
        return True

    # ---- buzzer
    def beep(self, duration: float = 0.1) -> None:
        self._pin.output_with_state(False)        # LOW = sound
        time.sleep(duration)
        self._pin.state = True
        self._to_input()

    def beep_pattern(self, count: int, on: float = 0.1, off: float = 0.1) -> None:
        for i in range(count):
            self.beep(on)
            if i < count - 1:
                time.sleep(off)

    def on_close(self) -> None:
        self._pin.close()
