"""NewUi 3 robot face: mood and motion model (pure Python, no drawing).

Display only. The renderer reads the UI's SubMaster and passes a FaceInputs each
frame; nothing here reads or changes driving control.

The motion follows the web prototype: every part of the face is a damped spring
with its own stiffness, so eyes lead and the head follows; blinks, saccades, slow
drift, breathing and idle habits keep it alive between events; car events trigger
short choreographies (anticipation, action, settle).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np


def clamp(v: float, a: float, b: float) -> float:
  return a if v < a else b if v > b else v


def lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * t


def fract(x: float) -> float:
  return x - math.floor(x)


def ease_out_cubic(x: float) -> float:
  return 1 - (1 - x) ** 3


def ease_in_out_sine(x: float) -> float:
  return -(math.cos(math.pi * x) - 1) / 2


def ease_out_back(x: float) -> float:
  c1 = 1.70158
  c3 = c1 + 1
  return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def _rnd(n: float) -> float:
  return fract(math.sin(n * 12.9898 + 78.233) * 43758.5453) * 2 - 1


def vnoise(x: float, seed: float) -> float:
  i = math.floor(x)
  f = x - i
  u = f * f * (3 - 2 * f)
  return lerp(_rnd(i + seed * 101.3), _rnd(i + 1 + seed * 101.3), u)


def fbm(x: float, seed: float) -> float:
  return vnoise(x, seed) * 0.62 + vnoise(x * 2.13, seed + 7) * 0.27 + vnoise(x * 4.71, seed + 13) * 0.11


class Spring:
  __slots__ = ('v', 't', 'vel', 'k', 'c')

  def __init__(self, v: float, k: float, zeta: float):
    self.v = v
    self.t = v
    self.vel = 0.0
    self.k = k
    self.c = 2 * math.sqrt(k) * zeta

  def step(self, dt: float) -> None:
    a = self.k * (self.t - self.v) - self.c * self.vel
    self.vel += a * dt
    self.v += self.vel * dt


# Expression targets. Unlisted keys fall back to BASE.
BASE = {'w': 96.0, 'h': 112.0, 'up': 0.0, 'tilt': 0.0, 'low': 0.0, 'smile': 0.0, 'sag': 0.0, 'asym': 0.0,
        'blush': 0.0, 'mO': 0.0, 'green': 0.0, 'amber': 0.0, 'flash': 0.0}
MOODS: dict[str, dict[str, float]] = {
  'idle': {},
  'happy': {'w': 108, 'h': 96, 'low': 0.46, 'smile': 1, 'blush': 0.9, 'green': 0.7},
  'focus': {'w': 100, 'h': 90, 'up': 0.12, 'tilt': 0.14, 'low': 0.07, 'smile': 0.25, 'green': 0.9},
  'surprised': {'w': 120, 'h': 146, 'mO': 0.9, 'flash': 1},
  'worried': {'w': 90, 'h': 102, 'up': 0.26, 'tilt': -0.68, 'low': 0.09, 'smile': -0.3, 'amber': 1},
  'sigh': {'w': 98, 'h': 98, 'up': 0.38, 'tilt': -0.12, 'low': 0.1, 'sag': 0.3, 'mO': 0.32},
  'drowsy': {'w': 102, 'h': 104, 'up': 0.55, 'tilt': 0.04, 'low': 0.12, 'smile': 0.1, 'sag': 0.5},
  'asleep': {'w': 104, 'h': 104, 'up': 1, 'low': 0.1, 'smile': 0.2, 'sag': 0.4},
  'curious': {'w': 98, 'h': 112, 'up': 0.04, 'tilt': 0.1, 'asym': 0.18},
  'contact': {'w': 100, 'h': 104, 'up': 0.06, 'low': 0.16, 'smile': 0.45, 'blush': 0.35},
  'think': {'w': 96, 'h': 108, 'tilt': -0.14, 'low': 0.05, 'asym': -0.06},
  'vibe': {'w': 102, 'h': 100, 'up': 0.1, 'low': 0.22, 'smile': 0.55, 'blush': 0.3},
  'shy': {'w': 94, 'h': 100, 'up': 0.2, 'tilt': -0.15, 'low': 0.14, 'smile': 0.4, 'blush': 0.85},
  'bored': {'w': 98, 'h': 100, 'up': 0.32, 'tilt': 0.05, 'low': 0.06, 'sag': 0.3},
  'ah': {'w': 100, 'h': 96, 'up': 0.4, 'tilt': -0.2, 'low': 0.12, 'mO': 0.35},
  'stretch': {'w': 106, 'h': 118, 'up': 0.97, 'low': 0.1, 'smile': 0.7, 'mO': 0.5},
}
# Engaged or not, these moods keep the driving tint.
KEEP_GREEN = ('vibe', 'contact', 'shy')
IDLE_LIKE = ('idle', 'focus', 'curious', 'contact', 'think', 'vibe', 'shy', 'bored')
DM_ALERTS = ('driverDistracted', 'driverUnresponsive', 'promptDriverDistracted', 'promptDriverUnresponsive',
             'preDriverDistracted', 'preDriverUnresponsive')


@dataclass
class FaceInputs:
  session: int = 0
  engaged: bool = False
  left_blinker: bool = False
  right_blinker: bool = False
  steer_deg: float = 0.0
  v_ego: float = 0.0
  a_ego: float = 0.0
  standstill: bool = False
  dm_alert: bool = False
  fcw: bool = False
  night: bool = False
  lane_phase: int = 0         # 0 off, 1 lane change starting, 2 finishing
  lane_dir: int = 0           # -1 left, 1 right
  blindspot_left: bool = False
  blindspot_right: bool = False
  lead_d: float = 0.0         # lead distance, m (0 = no lead)
  lead_v: float = 0.0         # lead's own speed, m/s
  cam_dist: float = 0.0       # speed camera ahead, m (0 = none)
  cam_limit: float = 0.0      # its limit, km/h
  route_left: float = 0.0     # navigation distance to the goal, m (0 = no route)
  steer_sat: bool = False     # steerSaturated is up
  cutin: bool = False         # radarCutin is up
  gas: bool = False
  lat_accel: float = 0.0      # m/s², + turning left


@dataclass
class Particle:
  kind: str
  x: float
  y: float
  vx: float = 0.0
  vy: float = 0.0
  life: float = 1.0
  age: float = 0.0
  size: float = 1.0
  phase: float = 0.0
  delay: float = 0.0
  direction: int = 1


@dataclass
class _Timer:
  at: float
  fn: Callable[[], None]
  token: int | None


@dataclass
class _Blink:
  depth: float
  speed: float
  double: bool
  t: float = 0.0


@dataclass
class _Path:
  t0: float
  dur: float
  f: Callable[[float], tuple[float, float]]
  head: float = 15.0
  done: Callable[[], None] | None = None


@dataclass
class _Osc:
  prop: str
  amp: float
  freq: float
  decay: float
  t: float = 0.0


def eye_span_arrays(cx: float, cy: float, w: float, h: float, n: float, up: float, low: float, tilt: float, smile: float,
                    sag: float, side: int, cols: int = 36):
  """A superellipse eye cut by the upper and lower lids, as vertical spans over cols + 1 columns.

  Returns arrays (x, top, bottom, top_is_lid, visible). Both lids are quadratic curves whose
  control x is the eye's centre, so their height is a simple function of x.
  """
  x0, y0 = cx - w / 2, cy - h / 2
  xl, xr = x0 - 16, x0 + w + 16
  base = y0 - 3 + up * (h + 6)
  t = tilt * h * 0.32
  y_left, y_right = (base - t, base + t) if side < 0 else (base + t, base - t)
  y_mid = (y_left + y_right) / 2 + sag * h * 0.12
  lower_base = y0 + h + 3 - low * h
  lower_mid = lower_base - smile * h * 0.56
  # columns bunch up toward the sides, where the outline is steep, so the rounded corners stay smooth
  x = x0 + w * (1 - np.cos(np.linspace(0.0, math.pi, cols + 1))) / 2
  u = (x - xl) / (xr - xl)
  upper = (1 - u) ** 2 * y_left + 2 * (1 - u) * u * y_mid + u ** 2 * y_right
  lower = (1 - u) ** 2 * lower_base + 2 * (1 - u) * u * lower_mid + u ** 2 * lower_base
  r = np.minimum(1.0, np.abs((x - cx) / (w / 2)))
  half = (h / 2) * np.maximum(0.0, 1 - r ** n) ** (1 / n)
  top = np.maximum(cy - half, upper)
  bottom = np.minimum(cy + half, lower)
  return x, top, bottom, upper > cy - half, bottom > top + 0.3


def eye_spans(cx: float, cy: float, w: float, h: float, n: float, up: float, low: float, tilt: float, smile: float,
              sag: float, side: int, cols: int = 36) -> list[tuple[float, float, float, bool]]:
  """eye_span_arrays as (x, top, bottom, top_is_lid) tuples, left to right, visible columns only."""
  x, top, bottom, lid, keep = eye_span_arrays(cx, cy, w, h, n, up, low, tilt, smile, sag, side, cols)
  return [(float(a), float(b), float(c), bool(d)) for a, b, c, d, k in zip(x, top, bottom, lid, keep, strict=True) if k]


class FaceModel:
  DROWSY_AFTER_S = 45.0       # stopped this long: drowsy, then asleep
  HARD_BRAKE = -3.5           # m/s², or a forward collision warning
  BRAKE_COOLDOWN_S = 3.0
  CAMERA_NEAR_M = 300.0       # react once per speed camera, this far out
  CAMERA_OVER_KPH = 5.0       # this far over its limit is speeding
  ARRIVE_M = 30.0
  CURVE_ACCEL = 2.3           # m/s² of lateral acceleration, held CURVE_HOLD_S
  CURVE_HOLD_S = 0.8
  # Who may cut in: a reaction runs only when nothing at least as important is still playing.
  PRIO = {'brake': 5, 'dm': 5, 'engage': 3, 'disengage': 3, 'dm_clear': 3, 'go': 2, 'sleepy': 0,
          'strain': 4, 'bsm': 4, 'cutin': 3, 'camera_fast': 2, 'lead_go': 2, 'arrive': 2,
          'hello': 1, 'lane_go': 1, 'lane_done': 1, 'camera': 1, 'gas': 1, 'curve': 1}
  COOLDOWN = {'strain': 6.0, 'bsm': 4.0, 'cutin': 6.0, 'lead_go': 10.0, 'gas': 10.0, 'curve': 40.0}
  WAKES = ('lead_go',)  # the only event that wakes it from sleep

  def __init__(self, rng: random.Random | None = None):
    self.rng = rng or random.Random()
    self.P: dict[str, Spring] = {}

    def d(name, v, k, zeta):
      self.P[name] = Spring(v, k, zeta)
    for name, v, k, z in (('w', 96, 170, 0.5), ('h', 112, 170, 0.46), ('up', 0, 240, 0.78), ('tilt', 0, 160, 0.72),
                          ('low', 0, 210, 0.76), ('smile', 0, 180, 0.64), ('sag', 0, 110, 0.8), ('asym', 0, 110, 0.68),
                          ('blush', 0, 26, 1), ('mO', 0, 190, 0.58), ('green', 0, 18, 1), ('amber', 0, 24, 1),
                          ('flash', 0, 80, 1), ('bright', 1, 12, 1),
                          ('gxL', 0, 760, 0.58), ('gyL', 0, 760, 0.6), ('gxR', 0, 690, 0.58), ('gyR', 0, 690, 0.6),
                          ('wL', 0, 320, 0.8), ('wR', 0, 320, 0.8),
                          ('hx', 0, 52, 0.62), ('hy', 0, 72, 0.4), ('rot', 0, 56, 0.55), ('pitch', 0, 90, 0.45),
                          ('ex', 0, 58, 0.55), ('ey', 0, 66, 0.5), ('er', 0, 56, 0.5), ('steer', 0, 9, 1),
                          ('p_shades', 0, 140, 0.55), ('p_quest', 0, 140, 0.55), ('p_bulb', 0, 140, 0.55),
                          ('p_nightcap', 0, 140, 0.55)):
      d(name, float(v), k, z)
    self.clock = 0.0
    self.session: int | None = None
    self.parts: list[Particle] = []
    self._timers: list[_Timer] = []
    self._token = 0
    self._oscs: list[_Osc] = []
    self.reset()

  # ---- state ------------------------------------------------------------------------
  def reset(self) -> None:
    for name in ('up', 'tilt', 'low', 'smile', 'sag', 'asym', 'blush', 'mO', 'green', 'amber', 'flash',
                 'ex', 'ey', 'er', 'pitch', 'wL', 'wR', 'p_shades', 'p_quest', 'p_bulb', 'p_nightcap'):
      self.P[name].t = 0.0
    self.mood = 'idle'
    self._apply_mood('idle')
    self.engaged = False
    self.signal = 0
    self.steer = 0.0
    self.night = False
    self.dm = False
    self.tremble = 0.0
    self.sleep_stage = 0
    self.attn = 1.0
    self.idle_for = 0.0
    self.stopped_for = 0.0
    self.squeeze = False
    self.shake = 0.0
    self.event_until = 0.0
    self.prio = 0
    self.last_event: str | None = None
    self.cooldown: dict[str, float] = {}
    self.brake_until = 0.0
    self.lane_phase = 0
    self.bsm_side = 0
    self.lead_move_for = 0.0
    self.cam_armed = True
    self.route_armed = True
    self.route_near_at = -1e9
    self.curve_for = 0.0
    self.steer_sat = False
    self.cutin = False
    self.gas = False
    self.star_from = 0.0
    self.star_until = 0.0
    self.eye_shape: str | None = None
    self.behavior: str | None = None
    self.behavior_timer = 4.0
    self.last_behaviors: list[str] = []
    self.bump_timer = 3.0
    self.blink: _Blink | None = None
    self.blink_timer = 1.4
    self.blink_l = self.blink_r = 0.0
    self.target = (0.0, 0.0)
    self.gaze_timer = 0.6
    self.micro = (0.0, 0.0)
    self.micro_timer = 0.0
    self.path: _Path | None = None
    self.hold: tuple[float, float, float] | None = None
    self.check_timer = 2.0
    self.dm_look = 0
    self.breath_phase = 0.0
    self.breath = 0.0
    self.tick_phase = 0.0
    self.z_timer = 0.0
    self.parts.clear()
    self._timers.clear()
    self._oscs.clear()
    self._token += 1

  def _apply_mood(self, name: str) -> None:
    self.mood = name
    m = MOODS[name]
    for k, v in BASE.items():
      self.P[k].t = float(m.get(k, v))
    if getattr(self, 'engaged', False) and name in KEEP_GREEN:
      self.P['green'].t = 0.6

  def rest_mood(self) -> str:
    return 'focus' if self.engaged else 'idle'

  # ---- timers ------------------------------------------------------------------------
  def later(self, sec: float, fn: Callable[[], None], keep: bool = False) -> None:
    self._timers.append(_Timer(self.clock + sec, fn, None if keep else self._token))

  def _cancel(self) -> None:
    self._token += 1

  def _run_timers(self) -> None:
    i = 0
    while i < len(self._timers):
      tm = self._timers[i]
      if self.clock >= tm.at:
        self._timers.pop(i)
        if tm.token is None or tm.token == self._token:
          tm.fn()
      else:
        i += 1

  # ---- helpers -----------------------------------------------------------------------
  def kick(self, name: str, v: float) -> None:
    self.P[name].vel += v

  def osc(self, prop: str, amp: float, freq: float, decay: float) -> None:
    self._oscs.append(_Osc(prop, amp, freq, decay))

  def osc_offset(self, prop: str) -> float:
    return sum(o.amp * math.sin(o.t * o.freq * 2 * math.pi) * math.exp(-o.decay * o.t) for o in self._oscs if o.prop == prop)

  def spawn(self, **kw) -> None:
    self.parts.append(Particle(**kw))

  def stars(self, sec: float) -> None:
    """Star eyes for sec seconds; they pop in from now."""
    self.star_from = self.clock
    self.star_until = self.clock + sec

  def wake(self) -> None:
    self.attn = 1.0
    self.idle_for = 0.0

  def look(self, x: float, y: float, sec: float) -> None:
    self.hold = (x, y, self.clock + sec)

  def start_blink(self, depth: float = 1.0, speed: float = 1.0, double: bool = False) -> None:
    if self.blink is None:
      self.blink = _Blink(depth, speed * (0.86 + self.rng.random() * 0.28), double)

  def _sparkle(self, n: int) -> None:
    for i in range(n):
      s = 1 if i % 2 else -1
      a = -math.pi / 2 + s * (0.3 + self.rng.random() * 0.8)
      sp = 60 + self.rng.random() * 70
      self.spawn(kind='spark', x=s * (100 + self.rng.random() * 50), y=-46 - self.rng.random() * 14,
                 vx=math.cos(a) * sp, vy=math.sin(a) * sp, life=0.9 + self.rng.random() * 0.5,
                 size=5 + self.rng.random() * 6, phase=self.rng.random() * 6)

  def _note(self) -> None:
    self.spawn(kind='note', x=120 + self.rng.random() * 60, y=-50, vx=18 + self.rng.random() * 14,
               vy=-40 - self.rng.random() * 16, life=1.6, size=1 + self.rng.random() * 0.3, phase=self.rng.random() * 6)

  def _heart(self) -> None:
    self.spawn(kind='heart', x=(self.rng.random() - 0.5) * 140, y=-20, vx=(self.rng.random() - 0.5) * 24,
               vy=-46 - self.rng.random() * 20, life=1.4, size=6 + self.rng.random() * 4)

  def _sweat(self) -> None:
    self.spawn(kind='sweat', x=-186, y=-64, vy=18, life=2.2)

  # ---- inputs -> events --------------------------------------------------------------------
  def _inputs(self, dt: float, inp: FaceInputs) -> None:
    if inp.session != self.session:
      self.session = inp.session
      self.reset()
      self.react('hello')
    self.night = inp.night
    self.steer = clamp(inp.steer_deg, -90.0, 90.0)
    sig = (-1 if inp.left_blinker else 0) + (1 if inp.right_blinker else 0)   # hazards cancel out
    if sig != self.signal:
      self.signal = sig
      self._on_signal(sig)
    if inp.engaged != self.engaged:
      self.react('engage' if inp.engaged else 'disengage')
    if inp.dm_alert != self.dm:
      self.react('dm' if inp.dm_alert else 'dm_clear')
    hard = inp.fcw or inp.a_ego < self.HARD_BRAKE
    if hard and self.clock >= self.brake_until:
      self.brake_until = self.clock + self.BRAKE_COOLDOWN_S
      self.react('brake')
    stopped = inp.standstill or abs(inp.v_ego) < 0.1
    self.stopped_for = self.stopped_for + dt if stopped else 0.0
    if self.sleep_stage == 0 and self.stopped_for > self.DROWSY_AFTER_S and not self.dm and not self.signal:
      self.react('sleepy')
    elif self.sleep_stage and inp.v_ego > 0.5:
      self.react('go')
    self._driving_events(dt, inp, stopped)

  def _driving_events(self, dt: float, inp: FaceInputs, stopped: bool) -> None:
    # lane change: lean into it, then a proud little nod once it finishes (not when it is called off)
    if inp.lane_phase != self.lane_phase:
      if inp.lane_phase == 1 and self.lane_phase == 0 and inp.lane_dir:
        self.event('lane_go', inp.lane_dir)
      elif inp.lane_phase == 0 and self.lane_phase == 2:
        self.event('lane_done')
      self.lane_phase = inp.lane_phase
    # a car in the blind spot on the side we are signalling toward
    side = -1 if self.signal < 0 and inp.blindspot_left else 1 if self.signal > 0 and inp.blindspot_right else 0
    if side and not self.bsm_side:
      self.event('bsm', side)
    self.bsm_side = side
    # the car ahead pulls away while we wait
    moving_off = stopped and self.stopped_for > 1.5 and 0 < inp.lead_d < 35 and inp.lead_v > 1.0
    self.lead_move_for = self.lead_move_for + dt if moving_off else 0.0
    if self.lead_move_for >= 0.4 and self.lead_move_for - dt < 0.4:
      self.event('lead_go')
    # speed camera: one reaction per camera
    if inp.cam_dist <= 0 or inp.cam_dist > self.CAMERA_NEAR_M + 200:
      self.cam_armed = True
    elif self.cam_armed and inp.cam_dist <= self.CAMERA_NEAR_M and inp.cam_limit > 0 and inp.v_ego > 5:
      self.cam_armed = False
      self.event('camera_fast' if inp.v_ego * 3.6 > inp.cam_limit + self.CAMERA_OVER_KPH else 'camera')
    # steering at its limit, a car cutting in, the driver taking the gas
    if inp.steer_sat and not self.steer_sat and self.engaged:
      self.event('strain', 1 if self.steer >= 0 else -1)
    self.steer_sat = inp.steer_sat
    if inp.cutin and not self.cutin and inp.v_ego > 3:
      self.event('cutin')
    self.cutin = inp.cutin
    if inp.gas and not self.gas and self.engaged and inp.v_ego > 2:
      self.event('gas')
    self.gas = inp.gas
    # arriving: the route counts down to the goal, or ends close to it
    if inp.route_left > 0:
      if inp.route_left > 300:
        self.route_armed = True
      if inp.route_left <= 150:
        self.route_near_at = self.clock
      if self.route_armed and inp.route_left <= self.ARRIVE_M:
        self.route_armed = False
        self.event('arrive')
    elif self.route_armed and self.clock - self.route_near_at < 20:
      self.route_armed = False
      self.event('arrive')
    # a fast bend
    if inp.v_ego > 8 and abs(inp.lat_accel) > self.CURVE_ACCEL:
      self.curve_for += dt
      if self.curve_for >= self.CURVE_HOLD_S and self.curve_for - dt < self.CURVE_HOLD_S:
        self.event('curve', 1 if inp.lat_accel > 0 else -1)
    else:
      self.curve_for = 0.0

  def event(self, name: str, arg: float = 0.0) -> bool:
    """Run a driving-event reaction unless something more important is playing or it just ran."""
    prio = self.PRIO[name]
    if self.clock < self.cooldown.get(name, -1.0):
      return False
    if (self.dm and prio < 5) or (self.sleep_stage == 2 and name not in self.WAKES):
      return False
    if self.clock < self.event_until and prio <= self.prio:
      return False
    self.cooldown[name] = self.clock + self.COOLDOWN.get(name, 0.0)
    self.react(name, arg)
    return True

  def _on_signal(self, sig: int) -> None:
    self.gaze_timer = 0
    self.check_timer = 1.6
    self.wake()
    self.start_blink(speed=1.2)
    if sig:
      self.P['ex'].t = sig * 36
      self.P['er'].t = sig * 7
      self.kick('hx', sig * 160)
    else:
      self.P['ex'].t = 0
      self.P['er'].t = 0

  # ---- reactions: short choreographies -----------------------------------------------------
  def react(self, name: str, arg: float = 0.0) -> None:
    self._cancel()
    self.prio = self.PRIO.get(name, 1)
    self.last_event = name
    self.squeeze = False
    self.behavior = None
    self.path = None
    self.hold = None
    self.eye_shape = None
    for k in ('ey', 'wL', 'wR', 'pitch', 'p_shades', 'p_quest', 'p_bulb'):
      self.P[k].t = 0
    if not self.signal:
      self.P['ex'].t = 0
      self.P['er'].t = 0
    P = self.P
    if name == 'engage':
      self.engaged = True
      self.sleep_stage = 0
      self.wake()
      self.event_until = self.clock + 1.8
      self.blink_timer = max(self.blink_timer, 1.3)
      self.behavior_timer = 2.5
      P['h'].t, P['w'].t, P['pitch'].t = 102, 100, -0.02
      self.kick('hy', 120)                                    # small crouch

      def hop():
        self._apply_mood('happy')
        self.kick('hy', -250)
        P['pitch'].t = 0.015
        self._sparkle(5)
        self.stars(0.7)

      def land():
        self.kick('hy', 130)
        P['pitch'].t = 0
        self.osc('rot', 2.6, 2.4, 3)
      self.later(0.13, hop)
      self.later(0.48, land)
      self.later(1.05, lambda: self.start_blink(speed=0.5))
      self.later(1.65, lambda: (self._apply_mood('focus'), self.start_blink()))
    elif name == 'disengage':
      self.engaged = False
      self.wake()
      self.event_until = self.clock + 1.6
      self._apply_mood('sigh')
      P['ey'].t, P['pitch'].t = 12, -0.07
      self.kick('hy', 140)
      self.later(0.25, lambda: self.spawn(kind='sweat', x=-110, y=10, vy=28, life=1.3))
      self.later(0.9, lambda: (setattr(P['ey'], 't', 0), setattr(P['pitch'], 't', 0), self.kick('hy', -120)))
      self.later(1.3, lambda: (self._apply_mood(self.rest_mood()), self.start_blink(speed=0.8)))
    elif name == 'brake':
      self.sleep_stage = 0
      self.wake()
      self.event_until = self.clock + 2.3
      P['h'].t, P['w'].t, P['up'].t, P['pitch'].t = 66, 108, 0.34, -0.06
      self.kick('hy', 170)                                    # flinch back

      def pop():
        self._apply_mood('surprised')
        self.kick('hy', -440)
        P['pitch'].t = 0.17
        self.spawn(kind='bang', x=196, y=-84, life=1.4)
        self.shake = max(self.shake, 9.0)
        self.tremble = 0.6
      self.later(0.08, pop)
      self.later(0.55, lambda: setattr(self, 'tremble', 0.0))
      self.later(0.85, lambda: (setattr(P['pitch'], 't', -0.03), self.kick('hy', 60)))
      self.later(1.35, lambda: (self._apply_mood('sigh'), setattr(P['pitch'], 't', 0), self.kick('hy', 110)))
      self.later(2.1, lambda: (self._apply_mood(self.rest_mood()), self.start_blink()))
    elif name == 'dm':
      self.dm = True
      self.sleep_stage = 0
      self.wake()
      self._apply_mood('worried')
      self.tremble = 1.0
      self.dm_look = 0
      self.gaze_timer = 0
      P['ex'].t, P['er'].t = -28, -7
      self._sweat()
      self.start_blink()
      self.kick('hx', -160)
    elif name == 'dm_clear':
      self.dm = False
      self.tremble = 0.0
      self.wake()
      self.event_until = self.clock + 1.0
      self._apply_mood('sigh')
      self.kick('hy', -90)
      self.later(0.85, lambda: (self._apply_mood(self.rest_mood()), self.start_blink()))
    elif name == 'sleepy':
      self.sleep_stage = 1
      self.tremble = 0.0
      self._apply_mood('drowsy')

      def nod():
        if self.sleep_stage != 1:
          return
        P['up'].t, P['er'].t = 0.92, 6
        self.kick('hy', 220)
        self.later(0.65, lambda: self.sleep_stage == 1 and (setattr(P['up'], 't', 0.45), self.kick('hy', -300),
                                                              setattr(P['er'], 't', 0), self.start_blink(speed=1.4)))
        self.later(1.0, lambda: self.sleep_stage == 1 and setattr(P['up'], 't', 0.55))

      def yawn():
        if self.sleep_stage != 1:
          return
        P['h'].t, P['up'].t, P['mO'].t, P['ey'].t, P['pitch'].t = 112, 0.4, 0.35, -10, 0.05
        self.kick('hy', -90)
        self.later(0.42, lambda: self.sleep_stage == 1 and (setattr(P['mO'], 't', 1), setattr(P['up'], 't', 0.94),
                                                              setattr(P['w'], 't', 112), setattr(self, 'squeeze', True),
                                                              self.osc('rot', 4, 1.0, 1.4)))

        def close():
          if self.sleep_stage != 1:
            return
          self.squeeze = False
          P['mO'].t, P['up'].t, P['w'].t, P['h'].t, P['ey'].t, P['pitch'].t = 0.08, 0.6, 102, 104, 0, 0
          self.kick('hy', 120)
          self.start_blink(speed=0.6)
        self.later(1.75, close)

      def fall_asleep():
        if self.sleep_stage == 1:
          self.sleep_stage = 2
          self._apply_mood('asleep')
          self.z_timer = 0.8
      self.later(1.0, nod)
      self.later(3.2, yawn)
      self.later(5.4, nod)
      self.later(7.3, fall_asleep)
    elif name == 'go':
      self.sleep_stage = 0
      self.wake()
      self.event_until = self.clock + 2.6
      self._apply_mood('surprised')
      self.kick('hy', -480)
      self.spawn(kind='bang', x=196, y=-84, life=1.4)
      self.shake = max(self.shake, 5.0)
      self.later(0.32, lambda: (self._apply_mood('idle'), self.look(-0.8, 0, 0.26), setattr(P['ex'], 't', -34)))
      self.later(0.58, lambda: (self.look(0.8, 0, 0.26), setattr(P['ex'], 't', 34)))
      self.later(0.84, lambda: (self.look(0, 0, 0.2), setattr(P['ex'], 't', 0), self.osc('hx', 16, 3.4, 3.0),
                                self.osc('rot', 7, 3.4, 3.0)))
      self.later(1.2, lambda: self.start_blink(double=True, speed=1.2))
      self.later(1.55, lambda: (self._apply_mood('happy'), self.engaged or setattr(P['green'], 't', 0), self.kick('hy', -230)))
      self.later(2.4, lambda: self._apply_mood(self.rest_mood()))
    else:
      self._react_driving(name, arg)

  def _glad(self) -> None:
    """Happy face; the green driving tint only while engaged."""
    self._apply_mood('happy')
    if not self.engaged:
      self.P['green'].t = 0

  def _settle(self) -> None:
    """End of a driving-event reaction: back to the resting face (or the turn-signal lean)."""
    P = self.P
    for k in ('ey', 'pitch', 'wL', 'wR', 'p_quest'):
      P[k].t = 0
    P['ex'].t = self.signal * 36
    P['er'].t = self.signal * 7
    self.squeeze = False
    if not self.dm:
      self.tremble = 0.0
      if not self.sleep_stage:
        self._apply_mood(self.rest_mood())
    self.start_blink()

  def _react_driving(self, name: str, arg: float) -> None:
    P, L = self.P, self.later
    d = int(arg) or 1
    if name in self.WAKES or name in ('bsm', 'strain', 'cutin', 'arrive'):
      self.sleep_stage = 0
      self.stopped_for = 0.0      # count the stop afresh, or it dozes straight off again
      self.wake()
    if name == 'hello':
      # wakes up with the car: eyes open, a hop and a sparkle
      self.event_until = self.clock + 2.2
      P['up'].v = P['up'].t = 1.0
      L(0.35, lambda: (setattr(P['up'], 't', 0.35), self.kick('hy', -60)))
      L(0.75, lambda: (self._glad(), self.kick('hy', -230), self._sparkle(4)))
      L(1.1, lambda: self.kick('hy', 120))
      L(1.6, self._settle)
    elif name == 'lane_go':
      # "here we go": a determined squint, sliding over with speed lines trailing
      self.event_until = self.clock + 1.6
      P['w'].t, P['h'].t, P['up'].t, P['tilt'].t, P['low'].t = 112, 92, 0.16, 0.22, 0.08
      P['ex'].t, P['er'].t = d * 34, d * 8
      self.look(d * 0.95, -0.05, 1.1)
      self.kick('hx', d * 200)
      for i in range(4):
        L(i * 0.09, lambda i=i: self.spawn(kind='streak', x=-d * (150 + i * 14), y=-42 + i * 26, vx=-d * 240, life=0.5,
                                            size=1 + (i % 2) * 0.5, direction=-d))
      L(1.15, lambda: (setattr(P['ex'], 't', d * 14), setattr(P['er'], 't', 0), self.osc('rot', 3, 2.2, 3)))
      L(1.5, self._settle)
    elif name == 'lane_done':
      # proud of it: a happy squint and a double nod
      self.event_until = self.clock + 1.3
      self._glad()
      P['blush'].t = 0.4
      self.kick('hy', -150)
      L(0.2, lambda: self._sparkle(2))
      L(0.3, lambda: self.kick('hy', 110))
      L(0.48, lambda: self.kick('hy', -90))
      L(1.1, self._settle)
    elif name == 'bsm':
      # "whoa, someone's there": recoil from that side while staring at it, then a no-no shake
      self.event_until = self.clock + 2.0
      P['w'].t, P['h'].t, P['up'].t, P['tilt'].t, P['smile'].t, P['amber'].t = 112, 132, 0.0, -0.3, -0.2, 0.8
      P['ex'].t, P['er'].t = -d * 34, -d * 7
      self.look(d * 1.1, -0.05, 0.9)
      self.kick('hx', -d * 260)
      self.kick('hy', 120)
      self.spawn(kind='bang', x=d * 206, y=-70, life=1.3)
      L(0.45, lambda: (self.osc('hx', 12, 5.5, 3.5), self.osc('rot', 4, 5.5, 3.5)))
      L(1.6, lambda: (setattr(P['amber'], 't', 0), self._settle()))
    elif name == 'lead_go':
      # "oh, they're moving": wide eyes forward, then leaning in to go
      self.event_until = self.clock + 1.9
      P['w'].t, P['h'].t, P['up'].t = 110, 136, 0.0
      self.look(0.0, -0.3, 0.9)
      self.kick('hy', -260)
      self.spawn(kind='bang', x=196, y=-84, life=1.2)
      L(0.55, lambda: (setattr(P['pitch'], 't', 0.06), setattr(P['ey'], 't', -8), self.kick('hy', -120)))
      L(0.9, lambda: self.start_blink(double=True, speed=1.2))
      L(1.5, self._settle)
    elif name in ('camera', 'camera_fast'):
      # a speed camera: a wary glance up at it, the shutter flash; sweating if over its limit
      fast = name == 'camera_fast'
      self.event_until = self.clock + (2.6 if fast else 2.0)
      self.spawn(kind='cam', x=178, y=-86, life=1.8)
      self.look(0.7, -0.7, 0.45)
      P['up'].t, P['tilt'].t = 0.2, 0.18
      L(0.35, lambda: (setattr(P['flash'], 't', 0.9), self.start_blink(speed=1.6)))
      L(0.5, lambda: setattr(P['flash'], 't', 0))
      L(0.65, lambda: self.look(0.0, 0.0, 0.35))
      L(1.0, lambda: self.look(0.7, -0.7, 0.4))
      if fast:
        L(0.7, lambda: (self._apply_mood('worried'), self._sweat(), setattr(self, 'tremble', 0.5)))
        L(2.3, self._settle)
      else:
        L(1.6, self._settle)
    elif name == 'strain':
      # the turn is past the steering limit: screwed-up eyes, trembling, leaning into it
      self.event_until = self.clock + 2.0
      self.squeeze = True
      self.tremble = 0.8
      P['up'].t, P['amber'].t, P['pitch'].t = 0.97, 0.5, 0.05
      P['ex'].t, P['er'].t = d * 26, d * 9
      self._sweat()
      self.kick('hx', d * 200)
      L(1.5, lambda: (setattr(self, 'squeeze', False), setattr(self, 'tremble', 0.0), self._apply_mood('sigh'),
                      setattr(P['pitch'], 't', -0.03), self.kick('hy', 90)))
      L(2.0, self._settle)
    elif name == 'cutin':
      # someone cut in: a flinch back and a quick double blink
      self.event_until = self.clock + 1.4
      P['w'].t, P['h'].t, P['up'].t, P['pitch'].t = 104, 124, 0.0, -0.07
      self.kick('hy', 220)
      self.look(0.0, -0.1, 0.8)
      self.start_blink(speed=1.8, double=True)
      L(0.5, lambda: (setattr(P['pitch'], 't', 0), self.kick('hy', -90)))
      L(1.1, self._settle)
    elif name == 'gas':
      # the driver pressed the gas: a curious glance their way, "you've got it?"
      self.event_until = self.clock + 1.8
      self._apply_mood('curious')
      P['green'].t = 0.9 if self.engaged else 0
      P['er'].t, P['ex'].t, P['p_quest'].t = -8, -14, 1
      self.look(-0.65, 0.22, 1.0)
      L(1.1, lambda: (setattr(P['p_quest'], 't', 0), self.look(0, 0, 0.2), self.kick('hy', -90)))
      L(1.5, self._settle)
    elif name == 'arrive':
      # made it: star eyes, hops, sparkles and hearts
      self.event_until = self.clock + 3.1
      self._apply_mood('happy')
      self.stars(1.4)
      for i in range(3):
        L(i * 0.42, lambda: (self.kick('hy', -280), self._sparkle(4)))
      for i in range(4):
        L(0.3 + i * 0.35, self._heart)
      L(1.5, lambda: (self.osc('rot', 6, 2.2, 2.2), setattr(P['blush'], 't', 0.9)))
      L(2.9, self._settle)
    elif name == 'curve':
      # a fast bend: "wheee", swung toward the outside of the turn
      self.event_until = self.clock + 1.8
      P['w'].t, P['h'].t, P['mO'].t = 112, 128, 0.45
      P['ex'].t, P['er'].t = d * 30, d * 8
      self.kick('hx', d * 220)
      L(0.6, lambda: (self._glad(), self.kick('hy', -120)))
      L(1.5, self._settle)

  # ---- idle habits -------------------------------------------------------------------------
  def _end_behavior(self) -> None:
    self.behavior = None
    self.path = None
    self.hold = None
    self.eye_shape = None
    self.squeeze = False
    for k in ('ex', 'ey', 'er', 'wL', 'wR', 'pitch', 'p_shades', 'p_quest', 'p_bulb'):
      self.P[k].t = 0
    if self.signal:
      self._on_signal(self.signal)
    if self.sleep_stage == 0 and not self.dm:
      self._apply_mood(self.rest_mood())

  def _behaviors(self) -> dict[str, tuple[tuple[str, ...], float, float, Callable[[], None]]]:
    P, r = self.P, self.rng
    L = self.later

    def curious():
      self._apply_mood('curious')
      s = -1 if r.random() < 0.5 else 1
      P['er'].t, P['ex'].t = s * 11, s * 18
      self.kick('hy', -60)
      self.start_blink()
      L(0.25, lambda: setattr(P['p_quest'], 't', 1))
      L(1.8, self._end_behavior)

    def contact():
      self._apply_mood('contact')
      self.look(-0.06, 0.12, 1.9)
      L(0.35, lambda: self.start_blink(speed=0.45))
      L(2.0, self._end_behavior)

    def wink():
      side = 'wL' if r.random() < 0.5 else 'wR'
      self._apply_mood('contact')
      self.look(-0.05, 0.1, 1.3)
      P['er'].t = -5 if side == 'wL' else 5
      L(0.28, lambda: (setattr(P[side], 't', 1), self.kick('hy', -80)))
      L(0.68, lambda: setattr(P[side], 't', 0))
      L(1.4, self._end_behavior)

    def pursuit():
      s = -1 if r.random() < 0.5 else 1
      self.path = _Path(self.clock, 2.4 + r.random() * 0.8,
                        lambda k: (lerp(-0.95 * s, 0.95 * s, ease_in_out_sine(k)), -0.05 - math.sin(k * math.pi) * 0.08),
                        head=34, done=lambda: (self.start_blink(depth=0.6, speed=1.2), self._end_behavior()))

    def think():
      self._apply_mood('think')
      self.look(-0.45, -0.55, 1.3)
      P['er'].t, P['ex'].t = -7, -10

      def idea():
        P['p_bulb'].t = 1
        self._apply_mood('idle')
        P['h'].t, P['w'].t, P['er'].t, P['ex'].t = 120, 102, 0, 0
        self.kick('hy', -150)
        self.look(0.1, -0.2, 0.9)
      L(1.3, idea)
      L(2.5, lambda: (self.start_blink(), self._end_behavior()))

    def scan():
      self.path = _Path(self.clock, 2.8, lambda k: (math.sin(k * math.pi * 2) * 0.92, -0.05 + math.sin(k * math.pi * 4) * 0.05),
                        head=42, done=self._end_behavior)

    def mirror():
      self.look(-1.08, -0.08, 0.5)
      P['ex'].t = -22
      L(0.55, lambda: (self.look(1.08, -0.08, 0.5), setattr(P['ex'], 't', 22), self.start_blink(speed=1.3)))
      L(1.15, lambda: (self.look(0, 0, 0.2), setattr(P['ex'], 't', 0)))
      L(1.45, self._end_behavior)

    def peek():
      s = -1 if r.random() < 0.5 else 1
      P['ex'].t, P['er'].t = s * 125, s * 9
      self.look(s * 1.08, -0.05, 1.5)
      L(1.55, lambda: (setattr(P['ex'], 't', 0), setattr(P['er'], 't', 0), self.kick('hy', -120)))
      L(2.0, self._end_behavior)

    def shy():
      self._apply_mood('shy')
      self.look(0.55, 0.62, 1.7)
      P['er'].t, P['ex'].t = 10, 18
      L(1.9, self._end_behavior)

    def giggle():
      self._apply_mood('happy')
      if not self.engaged:
        P['green'].t = 0
      for i in range(3):
        L(i * 0.2, lambda: (self.kick('hy', -190), self.osc('rot', 3, 5, 4)))
      L(1.2, self._end_behavior)

    def hum():
      self._apply_mood('vibe')
      for i in range(6):
        sgn = -1 if i % 2 else 1
        L(i * 0.45, lambda sgn=sgn, i=i: (self.kick('hy', 80), setattr(P['ex'], 't', sgn * 12), setattr(P['er'], 't', sgn * 5),
                                           i % 2 == 0 and self._note()))
      L(2.9, self._end_behavior)

    def stretch():
      self._apply_mood('stretch')
      P['ey'].t, P['pitch'].t = -14, 0.07
      self.osc('rot', 3, 0.8, 1.2)
      L(1.35, lambda: (setattr(P['ey'], 't', 0), setattr(P['pitch'], 't', 0), self.kick('hy', 190), self._apply_mood('contact')))
      L(2.0, lambda: (self.start_blink(), self._end_behavior()))

    def roll():
      self._apply_mood('bored')
      self.path = _Path(self.clock, 1.3, lambda k: (math.sin(k * math.pi * 2) * 0.72, -math.cos(k * math.pi * 2) * 0.58),
                        head=10, done=self._sigh)

    def cool():
      P['p_shades'].t, P['er'].t = 1, -4
      L(0.5, lambda: self.kick('hy', -100))
      L(2.6, lambda: setattr(P['p_shades'], 't', 0))
      L(3.1, self._end_behavior)

    def love():
      self._apply_mood('contact')
      self.eye_shape = 'heart'
      P['blush'].t = 0.85
      self.kick('hy', -90)
      for i in range(4):
        L(i * 0.35, self._heart)
      L(1.9, lambda: setattr(self, 'eye_shape', None))
      L(2.1, self._end_behavior)

    def sneeze():
      self._apply_mood('ah')
      P['ey'].t, P['er'].t = -6, -4
      L(0.5, lambda: (setattr(P['up'], 't', 0.6), setattr(P['ey'], 't', -12), setattr(P['mO'], 't', 0.55), self.kick('hy', -50)))

      def choo():
        self.squeeze = True
        P['up'].t, P['mO'].t, P['ey'].t, P['pitch'].t = 0.97, 0.95, 8, 0.1
        self.kick('hy', 420)
        self.shake = max(self.shake, 5.0)
      L(0.95, choo)
      L(1.28, lambda: (setattr(self, 'squeeze', False), setattr(P['pitch'], 't', 0), setattr(P['ey'], 't', 0),
                       setattr(P['er'], 't', 0), self._apply_mood('shy'), self.osc('hx', 10, 3.6, 3.4), self.osc('rot', 4.5, 3.6, 3.4)))
      L(1.65, lambda: self.start_blink(double=True))
      L(2.5, self._end_behavior)

    # while engaged: small driving-companion habits
    def nod():
      P['up'].t = 0.2
      for i in range(2):
        L(0.12 + i * 0.3, lambda: self.kick('hy', 150))
      L(0.95, self._end_behavior)

    def sign():
      self.look(0.55, -0.6, 0.75)
      P['ex'].t, P['ey'].t, P['er'].t = 10, -5, 3
      L(0.8, lambda: (self.look(0.08, 0, 0.3), setattr(P['ex'], 't', 0), setattr(P['ey'], 't', 0), setattr(P['er'], 't', 0),
                      self.start_blink(speed=1.2)))
      L(1.25, self._end_behavior)

    def lane():
      self.path = _Path(self.clock, 2.4, lambda k: (math.sin(k * math.pi * 2) * 0.12, 0.22 + abs(math.sin(k * math.pi * 5)) * 0.18),
                        head=8, done=self._end_behavior)
      for i in range(5):
        L(i * 0.48, lambda: self.kick('hy', 45))

    def cruise():
      for i in range(4):
        sgn = -1 if i % 2 else 1
        L(i * 0.7, lambda sgn=sgn: (setattr(P['er'], 't', sgn * 5), setattr(P['ex'], 't', sgn * 10)))
      L(2.9, self._end_behavior)

    def proud():
      self._apply_mood('happy')
      self.kick('hy', -160)
      L(0.34, lambda: self.kick('hy', 90))
      L(1.1, self._end_behavior)

    def lockin():
      P['up'].t, P['tilt'].t, P['pitch'].t = 0.3, 0.3, 0.04
      self.look(0, -0.05, 1.6)
      L(1.7, self._end_behavior)

    def driver():
      self._apply_mood('contact')
      self.look(-0.6, 0.15, 0.9)
      P['ex'].t = -14
      L(0.4, lambda: self.start_blink(speed=0.6))
      L(1.0, lambda: (self.look(0.05, -0.05, 0.4), setattr(P['ex'], 't', 0)))
      L(1.5, self._end_behavior)

    I, F, B = ('idle',), ('focus',), ('idle', 'focus')
    return {
      # name: (modes, weight, min idle seconds, run)
      'curious': (I, 2.6, 0, curious), 'contact': (B, 3, 0, contact), 'wink': (B, 1.3, 0, wink),
      'pursuit': (I, 2.4, 0, pursuit), 'think': (I, 1.6, 0, think), 'scan': (B, 2, 0, scan), 'mirror': (B, 1.6, 0, mirror),
      'peek': (I, 1.3, 0, peek), 'shy': (I, 1.2, 0, shy), 'giggle': (I, 1.3, 0, giggle), 'hum': (B, 1.4, 6, hum),
      'stretch': (I, 1, 8, stretch), 'roll': (I, 1, 8, roll), 'sneeze': (I, 0.7, 5, sneeze), 'sigh': (I, 1, 12, self._sigh),
      'cool': (B, 1.5, 0, cool), 'love': (B, 1.3, 0, love),
      'nod': (F, 2, 0, nod), 'sign': (F, 1.6, 0, sign), 'lane': (F, 1.8, 0, lane), 'cruise': (F, 1.6, 0, cruise),
      'proud': (B, 1.2, 0, proud), 'lockin': (F, 1.4, 0, lockin), 'driver': (F, 1.4, 0, driver),
    }

  def _sigh(self) -> None:
    self._apply_mood('sigh')
    self.kick('hy', 90)
    self.P['pitch'].t = -0.03
    self.later(0.95, lambda: (setattr(self.P['pitch'], 't', 0), self._end_behavior(), self.start_blink()))

  def run_behavior(self, name: str) -> None:
    self.behavior = name
    self._behaviors()[name][3]()

  def _update_behaviors(self, dt: float) -> None:
    if (self.behavior or self.signal or self.dm or self.sleep_stage or self.clock < self.event_until or
        (self.hold and self.clock < self.hold[2]) or self.mood not in ('idle', 'focus')):
      return
    self.behavior_timer -= dt
    if self.behavior_timer > 0:
      return
    focus = self.mood == 'focus'
    self.behavior_timer = (2.5 + self.rng.random() * 3.5) if focus else (3.5 + self.rng.random() * 5)
    bored = 1 - self.attn
    choices = []
    for name, (modes, w, min_idle, _) in self._behaviors().items():
      if ('focus' if focus else 'idle') in modes and name not in self.last_behaviors and self.idle_for >= min_idle:
        choices.append((name, w * (1 + bored * 1.4 if name in ('sigh', 'roll', 'stretch', 'hum') else 1)))
    if not choices:
      return
    total = sum(w for _, w in choices)
    pick = self.rng.random() * total
    name = choices[-1][0]
    for n, w in choices:
      pick -= w
      if pick <= 0:
        name = n
        break
    self.last_behaviors = [name, *self.last_behaviors][:3]
    self.run_behavior(name)
    if name not in ('sigh', 'roll', 'stretch', 'hum'):
      self.attn = min(1.0, self.attn + 0.22)

  # ---- blink, gaze, breath, particles ------------------------------------------------------
  @staticmethod
  def _blink_shape(b: _Blink, t: float) -> float:
    if t <= 0:
      return 0.0
    close, hold, opening = 0.07 / b.speed, (0.01 if b.depth < 1 else 0.03) / b.speed, 0.16 / b.speed
    if t < close:
      return b.depth * (t / close) ** 2
    if t < close + hold:
      return b.depth
    if t < close + hold + opening:
      return b.depth * (1 - ease_out_cubic((t - close - hold) / opening))
    return -1.0

  def _update_blink(self, dt: float) -> None:
    if self.blink is None:
      self.blink_l = self.blink_r = 0.0
      if self.sleep_stage == 2 or self.squeeze:
        return
      self.blink_timer -= dt * (0.6 + self.attn * 0.4)
      if self.blink_timer <= 0:
        r = self.rng.random()
        if self.sleep_stage == 1:
          self.start_blink(speed=0.4)
        elif self.mood in ('focus', 'contact') and r < 0.14:
          self.start_blink(speed=0.48)
        elif r < 0.16:
          self.start_blink(double=True)
        elif r < 0.3:
          self.start_blink(depth=0.55, speed=1.2)
        else:
          self.start_blink()
        self.blink_timer = (2.4 + self.rng.random() * 2.6) if self.sleep_stage == 1 else (1.8 + self.rng.random() * 4.4)
      return
    b = self.blink
    b.t += dt
    left, right = self._blink_shape(b, b.t), self._blink_shape(b, b.t - 0.012)   # the right eye is 12 ms behind
    self.blink_l, self.blink_r = max(0.0, left), max(0.0, right)
    if left < 0 and right < 0:
      self.blink = None
      self.blink_l = self.blink_r = 0.0
      if b.double:
        self.blink_timer = 0.08

  def _update_gaze(self, dt: float) -> None:
    P, r = self.P, self.rng
    self.gaze_timer -= dt
    self.micro_timer -= dt
    lazy = 1 + (1 - self.attn) * 1.2
    head = 15.0
    if self.hold and self.clock < self.hold[2]:
      self.target = (self.hold[0], self.hold[1])
    elif self.path:
      k = (self.clock - self.path.t0) / self.path.dur
      head = self.path.head
      if k >= 1:
        done = self.path.done
        self.path = None
        self.target = (0.0, 0.0)
        self.gaze_timer = 0.8
        if done:
          done()
      else:
        self.target = self.path.f(k)
    elif self.signal:
      self.check_timer -= dt
      if self.check_timer <= 0:
        self.look(self.signal * 1.1, -0.05, 0.45)
        self.start_blink(speed=1.2)
        self.kick('hx', self.signal * 90)
        self.check_timer = 1.8 + r.random() * 1.2
      elif self.gaze_timer <= 0:
        self.target = (self.signal * (0.7 + r.random() * 0.14), (r.random() - 0.5) * 0.2)
        self.gaze_timer = 0.7 + r.random() * 0.9
    elif self.dm:
      if self.gaze_timer <= 0:
        self.dm_look = 1 - self.dm_look
        g = (r.random() + r.random() + r.random() - 1.5) / 1.5
        self.target = (-0.6 + g * 0.05, 0.14) if self.dm_look else (0.06 + g * 0.05, -0.06)
        if self.dm_look:
          self.kick('hx', -110)
          self.kick('hy', -50)
        self.gaze_timer = (0.7 + r.random() * 0.45) if self.dm_look else (0.42 + r.random() * 0.3)
        if r.random() < 0.35:
          self.start_blink(speed=1.25)
    elif self.sleep_stage == 2:
      self.target = (0.0, 0.18)
    elif self.gaze_timer <= 0:
      amp = (0.28 if self.mood == 'focus' else 0.16 if self.sleep_stage == 1 else 0.75) * (0.7 + self.attn * 0.3)
      g1 = (r.random() + r.random() + r.random() - 1.5) / 1.5
      g2 = (r.random() + r.random() + r.random() - 1.5) / 1.5
      wide = (-1 if r.random() < 0.5 else 1) * amp if r.random() < 0.22 else 0
      nx, ny = clamp(g1 * amp * 0.6 + wide, -1, 1), clamp(g2 * amp * 0.4, -0.7, 0.7)
      if math.hypot(nx - self.target[0], ny - self.target[1]) > 0.45 and r.random() < 0.42:
        self.start_blink(speed=1.1)
      self.target = (nx, ny)
      self.gaze_timer = ((2.4 if self.sleep_stage == 1 else 1.0) + r.random() * 2.6) * lazy
    if self.micro_timer <= 0:
      self.micro = ((r.random() - 0.5) * 0.045, (r.random() - 0.5) * 0.035)
      self.micro_timer = 0.2 + r.random() * 0.6
    drift = (fbm(self.clock * 0.55, 4) * 0.012, fbm(self.clock * 0.5, 5) * 0.01)
    steer_bias = (P['steer'].v / 90 * 0.62 if not self.signal and not self.dm and not self.hold and not self.path
                  and self.mood in ('focus', 'idle') else 0.0)
    tx = clamp(self.target[0] + self.micro[0] + drift[0] + steer_bias, -1.15, 1.15)
    ty = clamp(self.target[1] + self.micro[1] + drift[1] + (0.22 if self.sleep_stage == 1 else 0), -1, 1)
    P['gxL'].t = P['gxR'].t = tx
    P['gyL'].t = P['gyR'].t = ty
    steer_inertia = clamp((self.steer - P['steer'].v) * -0.07, -8, 8)
    P['hx'].t = tx * head
    P['rot'].t = (tx * 3.1 + (-4 if self.mood == 'think' else 0) + (4 if self.sleep_stage else 0) + steer_inertia +
                  (P['steer'].v * 0.06 if self.engaged else 0))
    P['hy'].t = ty * 7 + (6 if self.sleep_stage == 1 else 9 if self.sleep_stage == 2 else 0) + (-4 if self.mood == 'think' else 0)

  def _update_breath(self, dt: float) -> None:
    period = (6.2 if self.sleep_stage == 2 else 5.2 if self.sleep_stage == 1 else 4.9 if self.night else 4.3)
    period += fbm(self.clock * 0.1, 9) * 0.6
    self.breath_phase = fract(self.breath_phase + dt / period)
    p = self.breath_phase
    b = ease_in_out_sine(p / 0.4) if p < 0.4 else 1 - ease_in_out_sine((p - 0.4) / 0.6)
    self.breath = (b - 0.5) * (1.8 if self.sleep_stage == 2 else 1)

  def _update_particles(self, dt: float) -> None:
    if self.sleep_stage == 2:
      self.z_timer -= dt
      if self.z_timer <= 0:
        self.spawn(kind='z', x=150 + self.rng.random() * 18, y=-42, vx=20, vy=-24, life=2.8,
                   size=13 + self.rng.random() * 5, phase=self.rng.random() * 6)
        self.z_timer = 1.25
    if self.tremble > 0.5 and self.rng.random() < dt * 0.7:
      self._sweat()
    if self.signal:
      prev = self.tick_phase
      self.tick_phase += dt * 1.42
      if math.floor(self.tick_phase) != math.floor(prev):
        for i in range(3):
          self.spawn(kind='chev', x=self.signal * (206 + i * 16), y=-8, vx=self.signal * 40, life=0.5, delay=i * 0.07,
                     direction=self.signal)
        self.kick('hy', 70)
    else:
      self.tick_phase = 0.0
    for p in self.parts:
      p.age += dt
      p.x += p.vx * dt
      p.y += p.vy * dt
      if p.kind == 'spark':
        p.vy += 110 * dt
        p.vx *= 0.985
    self.parts[:] = [p for p in self.parts if p.age < p.life]

  # ---- main --------------------------------------------------------------------------------
  def update(self, dt: float, inp: FaceInputs) -> None:
    dt = clamp(dt, 0.0, 0.05)
    self.clock += dt
    self._inputs(dt, inp)
    self._run_timers()
    if not self.signal and not self.dm and not self.sleep_stage and self.mood in IDLE_LIKE:
      self.idle_for += dt
      self.attn = max(0.35, self.attn - dt / 28)
    self.shake *= math.exp(-dt * 7)
    if self.engaged and not self.sleep_stage and self.clock >= self.event_until:
      self.bump_timer -= dt
      if self.bump_timer <= 0:
        self.kick('hy', 35 + self.rng.random() * 45)                 # the odd road bump
        self.bump_timer = 2 + self.rng.random() * 4
    self._update_blink(dt)
    self._update_behaviors(dt)
    self._update_gaze(dt)
    self._update_breath(dt)
    for o in self._oscs:
      o.t += dt
    self._oscs[:] = [o for o in self._oscs if math.exp(-o.decay * o.t) >= 0.02]
    self._update_particles(dt)
    P = self.P
    P['bright'].t = 0.62 if self.night else 1.0
    P['steer'].t = self.steer
    P['p_nightcap'].t = 1.0 if self.sleep_stage == 2 else 0.0
    steps = max(1, math.ceil(dt / (1 / 120)))     # 120 Hz is stable for the stiffest spring (k 760) and halves the cost
    for _ in range(steps):
      for s in P.values():
        s.step(dt / steps)
