"""NewUi=3 robot face model: eye geometry, car-state reactions, habits. Pure Python, no drawing."""
import math
import random

import pytest

from openpilot.selfdrive.ui.mici.onroad.robot_face_model import FaceInputs, FaceModel, eye_spans


def run(m, inp, sec, dt=0.05, each=None):
  for _ in range(int(round(sec / dt))):
    if each:
      each()
    m.update(dt, inp)


@pytest.fixture
def m():
  model = FaceModel(random.Random(7))
  model.update(0.05, FaceInputs(session=1))
  return model


def test_open_eye_fills_the_outline_and_a_shut_one_is_empty():
  spans = eye_spans(0, 0, 90, 100, 3.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1)
  assert len(spans) >= 30
  mid = min(spans, key=lambda s: abs(s[0]))
  assert mid[1] == pytest.approx(-50, abs=0.5) and mid[2] == pytest.approx(50, abs=0.5)
  assert not any(s[3] for s in spans)                       # nothing cut by a lid
  assert eye_spans(0, 0, 90, 100, 3.2, 1.0, 0.0, 0.0, 0.0, 0.0, -1) == []


def test_happy_eye_is_a_crescent_and_tilt_slopes_the_lid():
  spans = eye_spans(0, 0, 90, 100, 3.2, 0.0, 0.46, 0.0, 1.0, 0.0, 1)
  mid = min(spans, key=lambda s: abs(s[0]))
  edge = spans[2]
  assert mid[1] == pytest.approx(-50, abs=0.5)
  assert mid[2] < -10 and edge[2] > mid[2]                  # the lower lid bulges up in the middle
  worried = eye_spans(0, 0, 90, 100, 3.2, 0.3, 0.0, -0.7, 0.0, 0.0, -1)
  # left eye, worried: the outer (left) side is cut lower than the inner (right) side
  assert worried[3][1] > worried[-4][1]


def test_engage_hops_with_star_eyes_then_drives_green(m):
  inp = FaceInputs(session=1, engaged=True)
  run(m, inp, 0.3)
  assert m.mood == 'happy' and m.clock < m.star_until
  run(m, inp, 1.6)
  assert m.mood == 'focus' and m.P['green'].t == pytest.approx(0.9)
  inp.engaged = False
  run(m, inp, 0.2)
  assert m.mood == 'sigh'
  run(m, inp, 1.3)
  assert m.mood == 'idle' and m.P['green'].t == 0


@pytest.mark.parametrize('left,right,sign', [(True, False, -1), (False, True, 1), (True, True, 0)])
def test_turn_signal_looks_that_way(m, left, right, sign):
  inp = FaceInputs(session=1, left_blinker=left, right_blinker=right)
  run(m, inp, 1.0)
  assert m.signal == sign
  if sign:
    assert math.copysign(1, m.P['gxL'].v) == sign and abs(m.P['gxL'].v) > 0.4
    assert any(p.kind == 'chev' for p in m.parts) or m.tick_phase > 0


def test_driver_monitoring_alert_worries_until_cleared(m):
  inp = FaceInputs(session=1, dm_alert=True)
  run(m, inp, 1.0)
  assert m.mood == 'worried' and m.tremble == 1.0 and m.P['amber'].t == 1
  inp.dm_alert = False
  run(m, inp, 0.2)
  assert m.mood == 'sigh' and m.tremble == 0
  run(m, inp, 1.0)
  assert m.mood == 'idle'


def test_hard_braking_startles_once_per_cooldown(m):
  inp = FaceInputs(session=1, a_ego=-4.0, v_ego=10)
  run(m, inp, 0.15)
  assert m.mood == 'surprised' and any(p.kind == 'bang' for p in m.parts)
  first_until = m.brake_until
  run(m, inp, 1.0)
  assert m.brake_until == first_until                       # still braking, no new startle inside 3 s
  inp.a_ego, inp.fcw = 0.0, True
  run(m, inp, 2.5)
  assert m.brake_until > first_until                        # a forward collision warning also startles


def test_long_stop_dozes_off_and_pulling_away_wakes_it(m):
  inp = FaceInputs(session=1, standstill=True)
  run(m, inp, 44.0)
  assert m.sleep_stage == 0
  run(m, inp, 1.5)
  assert m.sleep_stage == 1 and m.mood == 'drowsy'
  run(m, inp, 8.0)
  assert m.sleep_stage == 2 and m.mood == 'asleep' and m.P['p_nightcap'].v > 0.5
  assert any(p.kind == 'z' for p in m.parts)
  inp.standstill, inp.v_ego = False, 2.0
  run(m, inp, 0.1)
  assert m.sleep_stage == 0 and m.mood == 'surprised'
  run(m, inp, 3.0)
  assert m.mood == 'idle'


def test_new_session_resets_and_night_dims(m):
  run(m, FaceInputs(session=1, engaged=True, dm_alert=True), 2.0)
  run(m, FaceInputs(session=2, night=True), 0.1)
  assert m.mood == 'idle' and not m.dm and m.P['bright'].t == 0.62


def test_every_habit_runs_and_nothing_goes_wild():
  model = FaceModel(random.Random(3))
  names = list(model._behaviors())
  for engaged in (False, True):
    for name in names:
      model.react('engage' if engaged else 'disengage')
      model.update(0.05, FaceInputs(session=1, engaged=engaged))
      run(model, FaceInputs(session=1, engaged=engaged), 2.5)
      model.run_behavior(name)
      run(model, FaceInputs(session=1, engaged=engaged), 4.0)
      for k, s in model.P.items():
        assert math.isfinite(s.v) and abs(s.v) < 2000, (name, k, s.v)
  assert model.behavior is None or model.behavior in names


def test_long_random_drive_picks_many_habits_without_errors():
  model = FaceModel(random.Random(11))
  inp = FaceInputs(session=1)
  seen = set()
  for i in range(int(900 / 0.05)):
    t = i * 0.05
    inp.engaged = 300 < t < 600
    inp.right_blinker = 450 < t < 455
    inp.steer_deg = 25 * math.sin(t / 3)
    inp.v_ego = 15.0
    model.update(0.05, inp)
    if model.behavior:
      seen.add(model.behavior)
  assert len(seen) >= 15
  assert all(math.isfinite(s.v) for s in model.P.values())
