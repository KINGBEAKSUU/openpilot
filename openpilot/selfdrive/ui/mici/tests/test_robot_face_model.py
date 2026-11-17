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


# ---- driving-event reactions -------------------------------------------------------------------
def ready(engaged=False, **kw):
  """A model past its session-start hello (and engage hop), plus the inputs it is running on."""
  model = FaceModel(random.Random(4))
  inp = FaceInputs(session=1, engaged=engaged, v_ego=kw.pop('v_ego', 15.0), **kw)
  run(model, inp, 4.0)
  model.last_event = None
  return model, inp


def test_session_start_says_hello():
  model = FaceModel(random.Random(1))
  model.update(0.05, FaceInputs(session=1))
  assert model.last_event == 'hello' and model.P['up'].t == 1.0          # eyes start shut
  run(model, FaceInputs(session=1), 1.0)
  assert model.mood == 'happy'
  run(model, FaceInputs(session=1), 1.0)
  assert model.mood == 'idle' and model.P['up'].t == 0


def test_lane_change_leans_in_then_nods_only_when_it_finishes():
  m, inp = ready(engaged=True)
  inp.right_blinker = True
  run(m, inp, 0.5)
  inp.lane_phase, inp.lane_dir = 1, 1
  run(m, inp, 0.1)
  assert m.last_event == 'lane_go' and m.P['ex'].t > 0 and any(p.kind == 'streak' for p in m.parts)
  run(m, inp, 2.0)
  inp.lane_phase = 2
  run(m, inp, 0.5)
  inp.lane_phase, inp.right_blinker = 0, False
  run(m, inp, 0.1)
  assert m.last_event == 'lane_done' and m.mood == 'happy'
  run(m, inp, 3.0)
  assert m.mood == 'focus'
  inp.lane_phase, inp.lane_dir = 1, -1                    # called off before it finished: no nod
  run(m, inp, 2.0)
  inp.lane_phase = 0
  run(m, inp, 0.2)
  assert m.last_event == 'lane_go'


def test_blind_spot_car_on_the_signalled_side_startles():
  m, inp = ready(engaged=True)
  inp.blindspot_right = True                              # not signalling toward it: nothing
  run(m, inp, 0.5)
  assert m.last_event is None
  inp.right_blinker = True
  run(m, inp, 0.1)
  assert m.last_event == 'bsm' and m.P['ex'].t < 0 and m.P['amber'].t > 0
  assert any(p.kind == 'bang' and p.x > 0 for p in m.parts)
  m.last_event = None
  inp.blindspot_right = False
  run(m, inp, 0.5)
  inp.blindspot_right = True                              # back again inside the cooldown
  run(m, inp, 0.5)
  assert m.last_event is None


def test_car_ahead_pulling_away_while_stopped_even_from_sleep():
  m, inp = ready(v_ego=0.0, standstill=True, lead_d=8.0)
  inp.lead_v = 2.0
  run(m, inp, 0.3)
  assert m.last_event is None                             # has to keep moving for 0.4 s
  run(m, inp, 0.2)
  assert m.last_event == 'lead_go' and m.P['h'].t > 130
  m, inp = ready(v_ego=0.0, standstill=True, lead_d=8.0)
  run(m, inp, 55.0)
  assert m.sleep_stage == 2
  inp.lead_v = 1.6
  run(m, inp, 0.5)
  assert m.last_event == 'lead_go' and m.sleep_stage == 0
  run(m, inp, 2.0)
  assert m.sleep_stage == 0                               # still stopped, but it does not doze straight off again


def test_speed_camera_reacts_once_and_sweats_when_speeding():
  m, inp = ready(engaged=True, v_ego=20.0, cam_limit=80.0)
  inp.cam_dist = 600.0
  run(m, inp, 0.2)
  assert m.last_event is None
  inp.cam_dist = 280.0
  run(m, inp, 0.1)
  assert m.last_event == 'camera' and any(p.kind == 'cam' for p in m.parts)
  m.last_event = None
  for d in (200.0, 100.0, 20.0):
    inp.cam_dist = d
    run(m, inp, 0.5)
  assert m.last_event is None
  inp.cam_dist = 0.0                                      # passed it
  run(m, inp, 3.0)
  inp.cam_dist, inp.v_ego = 250.0, 25.0                   # 90 km/h at an 80 camera
  run(m, inp, 1.0)
  assert m.last_event == 'camera_fast' and m.mood == 'worried' and any(p.kind == 'sweat' for p in m.parts)


def test_steering_limit_cut_in_and_gas_override():
  m, inp = ready(engaged=True, steer_deg=-20.0)
  inp.steer_sat = True
  run(m, inp, 0.1)
  assert m.last_event == 'strain' and m.squeeze and m.P['ex'].t < 0
  inp.cutin = True                                        # less important than the strain still playing: it waits
  run(m, inp, 0.5)
  assert m.last_event == 'strain' and 'cutin' in m.pending
  inp.steer_sat = inp.cutin = False
  run(m, inp, 2.0)
  assert m.last_event == 'cutin' and not m.squeeze         # and plays right after
  run(m, inp, 2.0)
  inp.gas = True
  run(m, inp, 0.1)
  assert m.last_event == 'gas' and m.P['p_quest'].t == 1
  m, inp = ready(engaged=False)                           # not driving it: no strain, no gas
  inp.steer_sat = inp.gas = True
  run(m, inp, 0.5)
  assert m.last_event is None


def test_gas_glance_never_hides_a_lane_change_or_a_bend():
  # replayed drives: a gas press just before a lane change and a fast bend took both reactions away
  m, inp = ready(engaged=True)
  inp.gas = True
  run(m, inp, 0.3)
  assert m.last_event == 'gas'
  inp.right_blinker = True
  run(m, inp, 0.3)
  inp.lane_phase, inp.lane_dir = 1, 1
  run(m, inp, 0.1)
  assert m.last_event == 'lane_go' and m.P['asym'].t == 0   # none of the curious glance's lopsided eyes left over
  m, inp = ready(engaged=True)
  inp.gas = True
  run(m, inp, 0.2)
  inp.right_blinker = True                                # gas while signalling is just the lane change: no glance
  m.last_event = None
  inp.gas = False
  run(m, inp, 0.2)
  inp.gas = True
  run(m, inp, 0.3)
  assert m.last_event is None
  m, inp = ready(engaged=True)
  inp.cutin = True
  run(m, inp, 0.1)
  inp.lat_accel = 3.0                                     # the bend starts during the cut-in flinch
  run(m, inp, 1.0)
  assert m.last_event == 'cutin'
  run(m, inp, 0.6)
  assert m.last_event == 'curve'                          # queued, then played


def test_cut_in_at_crawling_speed():
  m, inp = ready(engaged=True, v_ego=2.8)                 # 10 km/h, as in the replayed drive
  inp.cutin = True
  run(m, inp, 0.1)
  assert m.last_event == 'cutin'


def test_driver_alert_and_hard_braking_outrank_driving_events():
  m, inp = ready(engaged=True)
  inp.dm_alert = True
  run(m, inp, 0.5)
  inp.lane_phase, inp.lane_dir = 1, 1
  run(m, inp, 0.5)
  assert m.last_event == 'dm' and m.mood == 'worried'
  inp.dm_alert, inp.lane_phase = False, 0
  run(m, inp, 3.0)
  inp.a_ego = -4.5
  run(m, inp, 0.1)
  inp.a_ego, inp.cutin = 0.0, True
  run(m, inp, 0.5)
  assert m.last_event == 'brake'


def test_arrival_at_the_goal():
  m, inp = ready()
  for d in (1200.0, 500.0, 120.0):
    inp.route_left = d
    run(m, inp, 0.5)
  assert m.last_event is None
  inp.route_left = 25.0
  run(m, inp, 0.1)
  assert m.last_event == 'arrive' and m.star_from < m.clock < m.star_until
  assert m.star_from > m.clock - 0.2                      # the stars pop in now, not 0.7 s before they end
  m, inp = ready()                                        # the route ends close to the goal
  for d in (900.0, 140.0, 0.0):
    inp.route_left = d
    run(m, inp, 0.5)
  assert m.last_event == 'arrive'
  m, inp = ready()                                        # cancelled far from it
  for d in (900.0, 0.0):
    inp.route_left = d
    run(m, inp, 0.5)
  assert m.last_event is None


def test_fast_bend_has_to_last():
  m, inp = ready(engaged=True)
  inp.lat_accel = 3.0
  run(m, inp, 0.5)
  assert m.last_event is None
  run(m, inp, 0.5)
  assert m.last_event == 'curve' and m.P['ex'].t > 0        # a left bend swings it to the right
  m, inp = ready(engaged=True, v_ego=5.0, lat_accel=3.0)  # slow: just a corner
  run(m, inp, 2.0)
  assert m.last_event is None


@pytest.mark.parametrize('engaged', [False, True])
def test_every_driving_reaction_settles_back(engaged):
  names = [n for n in FaceModel.PRIO if n not in ('brake', 'dm', 'engage', 'disengage', 'dm_clear', 'go', 'sleepy')]
  m, inp = ready(engaged=engaged)
  for name in names:
    m.behavior_timer = 1e9                                # no idle habit afterwards, so the resting face shows
    m.react(name, -1)
    run(m, inp, 3.5)
    assert m.mood == m.rest_mood() and not m.squeeze and m.tremble == 0, name
    assert m.P['ex'].t == 0 and m.P['p_quest'].t == 0, name
    assert all(math.isfinite(s.v) for s in m.P.values()), name
