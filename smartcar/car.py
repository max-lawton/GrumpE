"""The whole car as one object."""
from __future__ import annotations

from .actuators import DriveTrain, Fan, KeyBuzzer, RGBLed
from .base import Component
from .comms import BluetoothLink
from .sensors import (LightSensorPair, LineSensorArray, ObstacleSensorPair,
                      UltrasonicSensor)
from .servos import CameraGimbal, ScanServo


class Car(Component):
    """Owns every peripheral:

        car.drive        DriveTrain
        car.line         LineSensorArray
        car.obstacle     ObstacleSensorPair
        car.light        LightSensorPair
        car.ultrasonic   UltrasonicSensor
        car.scan_servo   ScanServo (carries the ultrasonic sensor)
        car.camera       CameraGimbal (.pan / .tilt)
        car.led          RGBLed
        car.key          KeyBuzzer
        car.fan          Fan
        car.bluetooth    BluetoothLink, or None unless bluetooth=True

    Use it as a context manager so the motors stop and the pins are released
    even if your code raises.
    """

    def __init__(self, *, bluetooth: bool = False):
        super().__init__("car")
        try:
            self.drive = self._adopt(DriveTrain())
            self.line = self._adopt(LineSensorArray())
            self.obstacle = self._adopt(ObstacleSensorPair())
            self.light = self._adopt(LightSensorPair())
            self.ultrasonic = self._adopt(UltrasonicSensor())
            self.scan_servo = self._adopt(ScanServo())
            self.camera = self._adopt(CameraGimbal())
            self.led = self._adopt(RGBLed())
            self.key = self._adopt(KeyBuzzer())
            self.fan = self._adopt(Fan())
            self.bluetooth = self._adopt(BluetoothLink()) if bluetooth else None
        except Exception:
            self.close()                          # don't leak half-built hardware
            raise

    def stop(self) -> None:
        self.drive.stop()
