"""AI car: a vision-language model drives the Yahboom car through tool calls.

Each loop:  camera frame + car state  ->  Ollama (gemma4 cloud)  ->  tool calls
            -> run them on the car -> send results back -> repeat until the
            model answers in plain text -> next loop.

    export OLLAMA_API_KEY=...        # never hard-code the key in this file
    python ai_car.py
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2

from smartcar import Car, RGBLed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = "6c03e4ce309b4829a7164235c578c39d.Bjs6Zyfute7snsthqmU9mnkR"
API_URL = "https://ollama.com/api/chat"
MODEL = "gemma4:31b-cloud"
REQUEST_TIMEOUT_S = 60
MAX_TOOL_ROUNDS = 6          # model<->tool round trips allowed per loop
LOOP_DELAY_S = 0.5           # breather between loops
ERROR_DELAY_S = 5            # wait after a network/API error

FRAME_WIDTH = 640            # frames are downscaled to this width before upload
JPEG_QUALITY = 70
FRAME_FLUSH = 2              # throw away buffered frames so the picture is fresh

# Motion limits. Speeds are 0..1 (the stock C code mostly used 150/255 = 0.6).
MIN_SPEED, MAX_SPEED, DEFAULT_SPEED = 0.25, 0.8, 0.5
MAX_MOVE_S = 3.0
MAX_BACK_S = 2.0             # no rear sensors, so reversing is kept short
MAX_BACK_SPEED = 0.5
TURN_SPEED = 0.6
# CALIBRATE ME: spin at TURN_SPEED for 1 s on your floor, measure the angle.
TURN_DEG_PER_SEC = 120.0
STOP_DISTANCE_CM = 15        # forward moves abort if something is closer
POLL_S = 0.05                # how often a move re-checks the sensors
SERVO_SETTLE_S = 0.4         # time for a servo to reach a new angle

SYSTEM = {"role": "system", "content": (
    "You are a grumpy robot car: sarcastic and put-upon, but secretly competent "
    "and careful.\n"
    "You are a small four-wheeled Raspberry Pi car with a camera on a pan/tilt "
    "mount (your eyes), an ultrasonic distance sensor on a scanning servo at the "
    "front, two infrared obstacle sensors, two light sensors, a line sensor under "
    "the nose, an RGB LED, a buzzer and a little fan. You act only by calling "
    "tools.\n\n"
    "Every loop you get your camera frame and a state report. Decide what to do, "
    "call tools, then finish with at most one or two short grumpy sentences. "
    "Actions matter more than talk.\n"
    "Rules:\n"
    "- Movement is timed and stops by itself. Prefer short moves (0.5-2 s), then "
    "look again: you get a fresh frame next loop.\n"
    "- If the picture and the distance sensor disagree, trust the distance sensor "
    "and stop.\n"
    "- You have no sensors at the back. Look around before reversing and keep it "
    "short.\n"
    "- If a move was stopped early by a sensor, do not repeat it. Turn or back "
    "away, then pick another route.\n"
    "- If you are unsure what you are seeing, stop and look around (set_camera, "
    "scan_surroundings) instead of guessing.\n"
    "- Only use the fan when there is a reason to blow air."
)}

msg_history = [
    SYSTEM,
    {"role": "user", "content": "Robot start."},
]

# Tool calls and tool results are messages too, so one loop is typically 3-6
# messages. Trimming drops whole loops (never part of one), see trim_history().
MSG_HISTORY_MAX = 30


def trim_history():
    """Drop the oldest whole loops until the history fits. SYSTEM always stays,
    and the newest loop is never cut, so a tool result can't be orphaned from
    the tool call that produced it."""
    while len(msg_history) > MSG_HISTORY_MAX:
        user_idx = [i for i, m in enumerate(msg_history) if i > 0 and m["role"] == "user"]
        if len(user_idx) < 2:
            break
        del msg_history[1:user_idx[1]]


# ---------------------------------------------------------------------------
# Hardware (created in main() so importing this file touches no GPIO)
# ---------------------------------------------------------------------------
car = None
camera = None

# What the car is doing, as far as the tools know. Servos are released after a
# move (to stop jitter), so their angles are tracked here instead of read back.
_state = {"motion": "stopped", "last_motion": None,
          "pan": 90, "tilt": 90, "scan_angle": 90}


def center_everything():
    """Camera and scan servo straight ahead. Called once at startup."""
    car.camera.center(wait=SERVO_SETTLE_S, release=True)
    car.scan_servo.look_front(wait=SERVO_SETTLE_S, release=True)
    _state.update(pan=90, tilt=90, scan_angle=90, motion="stopped")


# ---------------------------------------------------------------------------
# Helpers for the tools
# ---------------------------------------------------------------------------
def _num(value, lo, hi, name="value"):
    """Models sometimes send numbers as strings or go out of range: coerce and clamp."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return max(lo, min(hi, v))


def _aim_scan_front():
    """The ultrasonic sensor rides on the scan servo, so 'distance ahead' only
    means something when that servo faces front."""
    if _state["scan_angle"] != 90:
        car.scan_servo.look_front(wait=SERVO_SETTLE_S, release=True)
        _state["scan_angle"] = 90


def _front_hazard():
    """Reason string if something is in the way ahead, else None."""
    _aim_scan_front()
    blocked = [side for side, s in (("left", car.obstacle.left), ("right", car.obstacle.right))
               if s.blocked]
    if blocked:
        return f"infrared obstacle sensor ({' and '.join(blocked)}) triggered"
    d = car.ultrasonic.read()
    if d is not None and d < STOP_DISTANCE_CM:
        return f"ultrasonic sensor sees an obstacle {d:.0f} cm ahead"
    return None


def _run_motion(label, left, right, duration, guarded=False):
    """Drive for `duration` seconds, then ALWAYS stop. With guarded=True the
    front sensors are checked before starting and every POLL_S while moving."""
    reason = _front_hazard() if guarded else None
    moved_s = 0.0
    if reason is None:                           # path was clear: go
        _state["motion"] = label
        start = time.monotonic()
        try:
            car.drive.drive(left, right)
            while time.monotonic() - start < duration:
                if guarded:
                    reason = _front_hazard()
                    if reason:
                        break
                time.sleep(POLL_S)
        finally:
            car.drive.stop()
            _state["motion"] = "stopped"
            moved_s = time.monotonic() - start
    result = {"action": label, "moved_s": round(moved_s, 2),
              "stopped_early": reason is not None}
    if reason:
        result["reason"] = reason
    _state["last_motion"] = result
    return result


# ---------------------------------------------------------------------------
# Tool registry. @tool puts the schema (what the model sees) and the Python
# function (what runs) in one place, so they can't drift apart.
# ---------------------------------------------------------------------------
TOOL_FUNCS = {}
OLLAMA_TOOLS = []            # list of JSON-schema function specs for the API


def number(description, minimum=None, maximum=None, required=True):
    spec = {"type": "number", "description": description, "required": required}
    if minimum is not None:
        spec["minimum"] = minimum
    if maximum is not None:
        spec["maximum"] = maximum
    return spec


def string(description, enum=None, required=True):
    spec = {"type": "string", "description": description, "required": required}
    if enum:
        spec["enum"] = list(enum)
    return spec


def integer(description, minimum=None, maximum=None, required=True):
    spec = number(description, minimum, maximum, required)
    spec["type"] = "integer"
    return spec


def tool(description, **params):
    def register(fn):
        props, required = {}, []
        for pname, spec in params.items():
            spec = dict(spec)
            if spec.pop("required", True):
                required.append(pname)
            props[pname] = spec
        OLLAMA_TOOLS.append({"type": "function", "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        }})
        TOOL_FUNCS[fn.__name__] = fn
        return fn
    return register


# ---------------------------------------------------------------------------
# Tools: movement
# ---------------------------------------------------------------------------
@tool(
    "Drive straight forward for a fixed time, then stop automatically. How far "
    "you get depends on speed x time and the floor, so it is approximate: judge "
    "the result from the next camera frame. Safety: before and during the move "
    "the car checks the ultrasonic sensor and both infrared obstacle sensors and "
    "stops early if anything is closer than about 15 cm; the result says if that "
    "happened (stopped_early / reason). Those sensors only look straight ahead, "
    "so they cannot see things beside the car. Prefer short moves (0.5-2 s) and "
    "look again before the next one.",
    duration_seconds=number(f"How long to drive, 0.1 to {MAX_MOVE_S} seconds.", 0.1, MAX_MOVE_S),
    speed=number(f"Motor speed {MIN_SPEED} to {MAX_SPEED}; default {DEFAULT_SPEED}. "
                 "Use lower speeds near obstacles or in tight spaces.",
                 MIN_SPEED, MAX_SPEED, required=False),
)
def move_forward(duration_seconds, speed=DEFAULT_SPEED):
    d = _num(duration_seconds, 0.1, MAX_MOVE_S, "duration_seconds")
    s = _num(speed, MIN_SPEED, MAX_SPEED, "speed")
    result = _run_motion("forward", s, s, d, guarded=True)
    result["speed"] = s
    return result


@tool(
    "Drive straight backward for a fixed time, then stop automatically. WARNING: "
    "the car has NO sensors at the back, so nothing will stop it from hitting "
    "something behind it. Only reverse when you know the space behind is clear "
    "(for example after scanning or turning around to look), and keep it short. "
    "Useful for backing away after a move was stopped by an obstacle.",
    duration_seconds=number(f"How long to reverse, 0.1 to {MAX_BACK_S} seconds.", 0.1, MAX_BACK_S),
    speed=number(f"Motor speed {MIN_SPEED} to {MAX_BACK_SPEED}; default 0.4.",
                 MIN_SPEED, MAX_BACK_SPEED, required=False),
)
def move_backward(duration_seconds, speed=0.4):
    d = _num(duration_seconds, 0.1, MAX_BACK_S, "duration_seconds")
    s = _num(speed, MIN_SPEED, MAX_BACK_SPEED, "speed")
    result = _run_motion("backward", -s, -s, d)
    result["speed"] = s
    return result


@tool(
    "Drive forward while curving gently left or right (the inside wheels turn "
    "slower), for smooth steering around an obstacle or along a bend instead of "
    "stopping and turning in place. Same front-obstacle safety checks as "
    "move_forward. The path is approximate; confirm with the next camera frame.",
    direction=string("Which way to curve.", enum=["left", "right"]),
    duration_seconds=number(f"How long, 0.1 to {MAX_MOVE_S} seconds.", 0.1, MAX_MOVE_S),
    speed=number(f"Outer-wheel speed {MIN_SPEED} to {MAX_SPEED}; default {DEFAULT_SPEED}.",
                 MIN_SPEED, MAX_SPEED, required=False),
)
def curve_forward(direction, duration_seconds, speed=DEFAULT_SPEED):
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'")
    d = _num(duration_seconds, 0.1, MAX_MOVE_S, "duration_seconds")
    s = _num(speed, MIN_SPEED, MAX_SPEED, "speed")
    inner = s * 0.35
    left, right = (inner, s) if direction == "left" else (s, inner)
    result = _run_motion(f"curving {direction}", left, right, d, guarded=True)
    result["speed"] = s
    return result


def _turn(direction, degrees):
    deg = _num(degrees, 5, 360, "degrees")
    left, right = (-TURN_SPEED, TURN_SPEED) if direction == "left" else (TURN_SPEED, -TURN_SPEED)
    result = _run_motion(f"turning {direction}", left, right, deg / TURN_DEG_PER_SEC)
    result.update(requested_degrees=deg, approximate=True,
                  note="Timed turn with no gyro; check the next frame to confirm your heading.")
    return result


@tool(
    "Spin in place to the LEFT (counter-clockwise seen from above) by roughly the "
    "given angle, then stop. The turn is timed, not measured (no gyro), so expect "
    "maybe +/-20% error, more on carpet or with a weak battery: confirm your new "
    "heading from the next camera frame. 90 is about a quarter turn, 180 about "
    "facing back where you came from. The camera and ultrasonic sensor turn with "
    "the car. There are no obstacle checks while spinning, so don't turn right "
    "next to a wall.",
    degrees=number("Approximate angle to turn, 5 to 360 degrees.", 5, 360),
)
def turn_left(degrees):
    return _turn("left", degrees)


@tool(
    "Spin in place to the RIGHT (clockwise seen from above) by roughly the given "
    "angle, then stop. Timed, not measured (no gyro), so expect maybe +/-20% "
    "error: confirm your new heading from the next camera frame. 90 is about a "
    "quarter turn. No obstacle checks while spinning.",
    degrees=number("Approximate angle to turn, 5 to 360 degrees.", 5, 360),
)
def turn_right(degrees):
    return _turn("right", degrees)


@tool(
    "Immediately stop all wheels. Moves already stop by themselves when their time "
    "runs out, so this is mainly for holding still on purpose or as a safe default "
    "when you are unsure what is happening.",
)
def stop():
    car.drive.stop()
    _state["motion"] = "stopped"
    return {"stopped": True}


@tool(
    "Stay perfectly still for a while and do nothing (wheels stopped). Use it to "
    "let a moving thing (a person, a pet, a ball) get out of the way, or to pause "
    "when there is nothing useful to do.",
    seconds=number("How long to wait, 0.5 to 5 seconds.", 0.5, 5),
)
def wait(seconds):
    s = _num(seconds, 0.5, 5, "seconds")
    car.drive.stop()
    time.sleep(s)
    return {"waited_s": s}


# ---------------------------------------------------------------------------
# Tools: perception
# ---------------------------------------------------------------------------
@tool(
    "Measure the distance to the nearest object straight ahead of the car with the "
    "ultrasonic sensor (takes about half a second; the front scan servo is "
    "turned to face forward first if needed). Accurate for solid, flat surfaces "
    "roughly 3 cm to 4 m away. Soft, angled or very small things (cushions, "
    "curtains, chair legs) can reflect poorly. distance_cm is null when no echo "
    "came back, which means nothing was detected within range (open space), not "
    "that the sensor is broken. It cannot see to the sides: use scan_surroundings "
    "for that.",
)
def get_distance():
    _aim_scan_front()
    d = car.ultrasonic.read_filtered(samples=3, delay=0.03)
    if d is None:
        return {"distance_cm": None, "note": "no echo: nothing detected in range (or surface absorbed the sound)"}
    return {"distance_cm": round(d, 1)}


@tool(
    "Sweep the ultrasonic sensor right, front and left (the front scan servo "
    "swings 0, 90 and 180 degrees) and report a distance for each direction, "
    "then return it to face forward. Takes roughly 2-3 seconds and the car does "
    "not move. Use it before choosing which way to turn, when boxed in, or "
    "before reversing. null means no echo: nothing within range in that "
    "direction. 'clearest' is the direction with the most open space.",
)
def scan_surroundings():
    readings = {}
    for label, look in (("right", car.scan_servo.look_right),
                        ("front", car.scan_servo.look_front),
                        ("left", car.scan_servo.look_left)):
        look(wait=SERVO_SETTLE_S)
        d = car.ultrasonic.read_filtered(samples=3, delay=0.03)
        readings[label] = None if d is None else round(d, 1)
    car.scan_servo.look_front(wait=SERVO_SETTLE_S, release=True)
    _state["scan_angle"] = 90
    clearest = max(readings, key=lambda k: float("inf") if readings[k] is None else readings[k])
    return {"right_cm": readings["right"], "front_cm": readings["front"],
            "left_cm": readings["left"], "clearest": clearest,
            "note": "null = no echo (nothing detected in range)"}


@tool(
    "Point the camera (your eyes) by moving its pan/tilt servos, then hold there. "
    "pan: 0 to 180 degrees, 90 = straight ahead, larger values look further LEFT, "
    "smaller further RIGHT. tilt: 35 to 180 degrees, 90 = level, larger looks UP, "
    "smaller looks DOWN (35 is the lowest it can go). Give either or both. The "
    "new view appears in your NEXT camera frame, not in this tool's result. "
    "Looking around does not move the wheels; the ultrasonic sensor does not "
    "follow the camera. Re-centre (90, 90) before driving so the picture matches "
    "the direction of travel.",
    pan_degrees=number("Pan angle 0-180, 90 = straight ahead.", 0, 180, required=False),
    tilt_degrees=number("Tilt angle 35-180, 90 = level.", 35, 180, required=False),
)
def set_camera(pan_degrees=None, tilt_degrees=None):
    if pan_degrees is None and tilt_degrees is None:
        raise ValueError("give pan_degrees and/or tilt_degrees")
    if pan_degrees is not None:
        _state["pan"] = car.camera.pan.move_to(_num(pan_degrees, 0, 180, "pan_degrees"),
                                               wait=SERVO_SETTLE_S, release=True)
    if tilt_degrees is not None:
        _state["tilt"] = car.camera.tilt.move_to(_num(tilt_degrees, 35, 180, "tilt_degrees"),
                                                 wait=SERVO_SETTLE_S, release=True)
    return {"pan_degrees": _state["pan"], "tilt_degrees": _state["tilt"],
            "note": "look at your next camera frame to see the new view"}


@tool(
    "Read the digital short-range sensors that are not the camera or ultrasonic. "
    "obstacle_ir: two infrared proximity sensors at the front corners; "
    "blocked=true means something is within a few cm to roughly 30 cm "
    "(adjustable on the hardware) in front of that corner. Good at catching things "
    "the ultrasonic sensor misses beside you. light: two light sensors; "
    "lit=true means that side sees light above its threshold, so you can steer "
    "toward a torch by turning toward the lit side. line: four downward sensors "
    "(outer_left, inner_left, inner_right, outer_right) as a string of 1/0, "
    "1 = black line under that sensor.",
)
def get_sensors():
    return {
        "obstacle_ir": {"left_blocked": car.obstacle.left.blocked,
                        "right_blocked": car.obstacle.right.blocked},
        "light": {"left_lit": car.light.left.lit, "right_lit": car.light.right.lit},
        "line": {"pattern": car.line.pattern(),
                 "order": "outer_left, inner_left, inner_right, outer_right; 1 = black line"},
    }


@tool(
    "Get a full status report of the car: what the wheels are doing, the last "
    "move's outcome (including whether a sensor stopped it early), camera pan/tilt, "
    "scan servo angle, distance ahead, infrared and light sensors, line sensor, "
    "LED colour and fan. You already receive this at the start of every loop; "
    "call it again only to check after something you did.",
)
def get_state():
    _aim_scan_front()
    d = car.ultrasonic.read_filtered(samples=3, delay=0.03)
    return {
        "motion": _state["motion"],
        "last_motion": _state["last_motion"],
        "wheels": {"left": round(car.drive.left.speed, 2), "right": round(car.drive.right.speed, 2)},
        "camera": {"pan_degrees": _state["pan"], "tilt_degrees": _state["tilt"]},
        "scan_servo_degrees": _state["scan_angle"],
        "distance_ahead_cm": None if d is None else round(d, 1),
        **get_sensors(),
        "led_rgb": list(car.led.rgb),
        "fan_on": car.fan.is_on,
    }


# ---------------------------------------------------------------------------
# Tools: expression
# ---------------------------------------------------------------------------
@tool(
    "Set the RGB LED on the car, which people nearby can see. Use it to show mood "
    "or intent (red = annoyed/danger, green = all clear, blue = thinking, yellow "
    "= caution, off = dark). Give either a named colour, or red/green/blue "
    "brightness values 0-255 for a custom colour.",
    color=string("Named colour.", enum=list(RGBLed.COLORS), required=False),
    red=integer("Red brightness 0-255 (custom colour).", 0, 255, required=False),
    green=integer("Green brightness 0-255 (custom colour).", 0, 255, required=False),
    blue=integer("Blue brightness 0-255 (custom colour).", 0, 255, required=False),
)
def set_led(color=None, red=None, green=None, blue=None):
    if color is not None:
        if color not in RGBLed.COLORS:
            raise ValueError(f"unknown colour {color!r}; choose from {list(RGBLed.COLORS)}")
        car.led.set_color(color)
    elif any(v is not None for v in (red, green, blue)):
        car.led.set(*(int(_num(v or 0, 0, 255, "rgb")) for v in (red, green, blue)))
    else:
        raise ValueError("give a colour name or red/green/blue values")
    return {"led_rgb": list(car.led.rgb)}


@tool(
    "Sound the onboard buzzer: beeps to get attention, express annoyance, or "
    "signal that you are about to move or have finished. Blocks while beeping.",
    count=integer("Number of beeps, 1 to 5.", 1, 5, required=False),
    duration_seconds=number("Length of each beep, 0.05 to 1 second; default 0.1.",
                            0.05, 1.0, required=False),
)
def beep(count=1, duration_seconds=0.1):
    n = int(_num(count, 1, 5, "count"))
    d = _num(duration_seconds, 0.05, 1.0, "duration_seconds")
    car.key.beep_pattern(n, on=d, off=0.1)
    return {"beeped": n}


@tool(
    "Run the small fan on the front for a few seconds, then switch it off "
    "automatically. It is a toy fire-extinguisher fan: it can blow out a candle "
    "or flutter paper at very short range, nothing more. Use it only when there "
    "is a clear reason (someone asks, or a small flame or light object is right "
    "in front of you); never as a default action.",
    duration_seconds=number("How long to run the fan, 0.5 to 5 seconds.", 0.5, 5),
)
def blow_fan(duration_seconds):
    d = _num(duration_seconds, 0.5, 5, "duration_seconds")
    car.fan.on()
    try:
        time.sleep(d)
    finally:
        car.fan.off()
    return {"fan_ran_s": d}


# ---------------------------------------------------------------------------
# Running tool calls
# ---------------------------------------------------------------------------
def execute_tool(name, args):
    """Run one tool call and return a JSON string for the model. Never raises:
    a bad call comes back as {"error": ...} so the model can correct itself."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name!r}", "available": sorted(TOOL_FUNCS)})
    if isinstance(args, str):                    # some servers send arguments as a JSON string
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": f"arguments for {name} were not valid JSON"})
    if not isinstance(args, dict):
        return json.dumps({"error": f"arguments for {name} must be an object"})
    try:
        result = fn(**args)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})
    except Exception as exc:                     # hardware fault etc: don't keep driving
        car.drive.stop()
        _state["motion"] = "stopped"
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Camera and model
# ---------------------------------------------------------------------------
def capture_frame():
    """Fresh JPEG from the camera as base64 (what Ollama's `images` field wants),
    or None if the camera gave nothing."""
    for _ in range(FRAME_FLUSH):                 # drain the driver's frame buffer
        camera.grab()
    ret, frame = camera.read()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def chat(messages):
    """One request to the Ollama API. Returns the assistant message dict."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": OLLAMA_TOOLS,
        "stream": False,
        "thinking": False,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())["message"]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama API returned HTTP {exc.code}: "
                           f"{exc.read().decode('utf-8', 'replace')[:300]}") from exc


def loop():
    """One full cycle: look, think, act."""
    car.drive.stop()                             # start every cycle from a known-safe state

    # Old frames are expensive and stale: only the newest message keeps its image.
    for m in msg_history:
        m.pop("images", None)
        m.pop("thinking", None)

    frame_b64 = capture_frame()
    state_json = json.dumps(get_state())

    user_msg = {"role": "user", "content": (
        "Run next loop.\n"
        f"Car state: {state_json}\n"
        + ("Current camera frame attached." if frame_b64 else
           "Camera frame unavailable this loop: rely on the sensors and be careful.")
    )}
    if frame_b64:
        user_msg["images"] = [frame_b64]
    msg_history.append(user_msg)
    trim_history()

    for _ in range(MAX_TOOL_ROUNDS):
        message = chat(msg_history)
        msg_history.append(message)
        if message.get("content"):
            print(f"[car] {message['content'].strip()}")

        calls = message.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            fn = call.get("function", {})
            name, args = fn.get("name", ""), fn.get("arguments") or {}
            result = execute_tool(name, args)
            print(f"  tool {name}({json.dumps(args)}) -> {result}")
            msg_history.append({"role": "tool", "tool_name": name, "content": result})
    else:
        print(f"[loop] still calling tools after {MAX_TOOL_ROUNDS} rounds; moving on")

    car.drive.stop()


def main():
    global car, camera
    if not API_KEY:
        sys.exit("Set your key first:  export OLLAMA_API_KEY=...")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        sys.exit("Could not open the camera (index 0).")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with Car() as car_:
        car = car_
        center_everything()
        try:
            while True:
                try:
                    loop()
                except (OSError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
                    car.drive.stop()
                    print(f"[error] {type(exc).__name__}: {exc}  (retrying in {ERROR_DELAY_S}s)")
                    time.sleep(ERROR_DELAY_S)
                time.sleep(LOOP_DELAY_S)
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            car.drive.stop()
            camera.release()


if __name__ == "__main__":
    main()