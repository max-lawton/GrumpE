"""Pin map for the Yahboom 4WD smart car, in BCM numbering (what gpiozero uses).

The original C programs used wiringPi numbers. Every pin below is written as
_w(<wiringPi number>) so it can be checked line by line against the Yahboom
sources. WPI_TO_BCM is the only table that has to be right.
"""

# wiringPi number -> BCM GPIO number (Raspberry Pi 40-pin header).
WPI_TO_BCM = {
    0: 17, 1: 18, 2: 27, 3: 22, 4: 23, 5: 24, 6: 25, 7: 4, 8: 2, 9: 3,
    10: 8, 11: 7, 12: 10, 13: 9, 14: 11, 15: 14, 16: 15,
    21: 5, 22: 6, 23: 13, 24: 19, 25: 26, 26: 12, 27: 16, 28: 20, 29: 21,
    30: 0, 31: 1,
}


def _w(wpi: int) -> int:
    return WPI_TO_BCM[wpi]


class Pins:
    # --- Motor driver (H-bridge): direction pins are digital, "pwm" sets speed
    LEFT_MOTOR_GO = _w(28)       # AIN2
    LEFT_MOTOR_BACK = _w(29)     # AIN1
    LEFT_MOTOR_PWM = _w(27)      # PWMA
    RIGHT_MOTOR_GO = _w(24)      # BIN2
    RIGHT_MOTOR_BACK = _w(25)    # BIN1
    RIGHT_MOTOR_PWM = _w(23)     # PWMB

    # --- Onboard button and buzzer share ONE pin (wiringPi 10).
    #     Button: LOW when pressed.  Buzzer: LOW = sound.
    KEY_BUZZER = _w(10)

    # --- 4-channel line tracker, ordered outer-left, inner-left,
    #     inner-right, outer-right. LOW = black line under the sensor.
    TRACK = (_w(9), _w(21), _w(7), _w(1))

    # --- Obstacle-avoidance IR sensors. LOW = obstacle.
    AVOID_LEFT = _w(26)
    AVOID_RIGHT = _w(0)

    # --- Light sensors (digital LDR modules). HIGH = light seen.
    LDR_LEFT = _w(11)
    LDR_RIGHT = _w(22)

    # --- Ultrasonic ranger. wiringPi 30/31 are BCM 0/1 (the HAT ID EEPROM
    #     pins), so keep i2c-0 disabled.
    ULTRA_ECHO = _w(30)
    ULTRA_TRIG = _w(31)

    # --- RGB LED (common cathode: HIGH = on)
    LED_R = _w(3)
    LED_G = _w(2)
    LED_B = _w(5)

    # --- Servos
    SERVO_FRONT = _w(4)          # pan servo carrying the ultrasonic sensor
    CAM_TILT = _w(13)            # camera up/down  (stock: ServoUpDownPin)
    CAM_PAN = _w(14)             # camera left/right (stock: ServoLeftRightPin)

    # --- Fire-extinguisher fan. Stock idles it HIGH, so LOW = running.
    OUTFIRE = _w(8)

    # --- Bluetooth UART
    SERIAL_PORT = "/dev/ttyAMA0"
    SERIAL_BAUD = 9600
