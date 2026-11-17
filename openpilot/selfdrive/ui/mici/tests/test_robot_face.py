"""NewUi=3 robot face renderer: car-state inputs and draw calls with a recording raylib stub."""
from collections import namedtuple
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

ONROAD = Path(__file__).parents[1] / 'onroad'


class SM(dict):
  def __init__(self):
    super().__init__(
      carState=NS(leftBlinker=False, rightBlinker=False, steeringAngleDeg=0.0, vEgo=10.0, aEgo=0.0, standstill=False,
                  leftBlindspot=False, rightBlindspot=False, gasPressed=False),
      selfdriveState=NS(enabled=False, alertType='', alertSound='none'), carControl=NS(latActive=False, actuators=NS(torque=0.0)),
      carOutput=NS(actuatorsOutput=NS(torque=0.0)),
      modelV2=NS(meta=NS(laneChangeState='off', laneChangeDirection='none')),
      radarState=NS(leadOne=NS(status=False, dRel=0.0, vLead=0.0)),
      controlsState=NS(curvature=0.0), carrotMan=NS(xSpdDist=0, xSpdLimit=0, nGoPosDist=0), onroadEvents=[])
    names = ('carState', 'selfdriveState', 'carControl', 'carOutput', 'modelV2', 'radarState', 'controlsState',
             'carrotMan', 'onroadEvents')
    self.alive = dict.fromkeys(names, True)
    self.valid = dict.fromkeys(names, True)
    self.recv_frame = dict.fromkeys(names, 11)


@pytest.fixture
def face(monkeypatch):
  calls = []
  Color = namedtuple('Color', 'r g b a')
  rl = NS(Color=Color, Rectangle=lambda *a: tuple(a), Vector2=lambda x, y: (x, y), WHITE=Color(255, 255, 255, 255),
          BLACK=Color(0, 0, 0, 255), RL_TRIANGLES=4,
          gen_image_gradient_radial=lambda *a: 'img', load_texture_from_image=lambda img: NS(width=128, height=128),
          measure_text_ex=lambda font, text, size, sp: NS(x=size * 0.6, y=size))
  for name in ('draw_rectangle_rec', 'unload_image', 'draw_texture_pro', 'rl_check_render_batch_limit', 'rl_begin',
               'rl_color4ub', 'rl_vertex2f', 'rl_end', 'draw_line_ex', 'draw_circle_v', 'draw_ellipse',
               'rl_disable_backface_culling', 'rl_enable_backface_culling', 'rl_push_matrix', 'rl_pop_matrix',
               'rl_translatef', 'rl_rotatef', 'rl_scalef', 'draw_rectangle_rounded', 'draw_rectangle_rounded_lines_ex',
               'draw_text_ex', 'draw_triangle_strip'):
    setattr(rl, name, lambda *a, name=name: calls.append((name, *a)))
  ui = NS(started_frame=10, sm=SM())
  monkeypatch.setitem(sys.modules, 'pyray', rl)
  monkeypatch.setitem(sys.modules, 'openpilot.selfdrive.ui.ui_state', NS(ui_state=ui))
  monkeypatch.setitem(sys.modules, 'openpilot.system.ui.lib.application', NS(gui_app=NS(font=lambda *_: 'font', target_fps=20),
                                                                              FontWeight=NS(BOLD='bold')))
  clock_spec = importlib.util.spec_from_file_location('_test_robot_clock', ONROAD / 'clock_face.py')
  clock_mod = importlib.util.module_from_spec(clock_spec)
  clock_spec.loader.exec_module(clock_mod)
  monkeypatch.setitem(sys.modules, 'openpilot.selfdrive.ui.mici.onroad.clock_face', clock_mod)
  spec = importlib.util.spec_from_file_location('_test_robot_face', ONROAD / 'robot_face.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  f = module.RobotFace()
  f._audio_chunks = lambda: []           # no microphone in tests
  return NS(f=f, ui=ui, calls=calls, rect=NS(x=0, y=0, width=536, height=240))


def names(calls):
  return [c[0] for c in calls]


def test_inputs_read_engagement_alerts_and_motion(face):
  sm = face.ui.sm
  inp = face.f._inputs()
  assert not inp.engaged and not inp.dm_alert and not inp.fcw and inp.v_ego == 10.0
  sm['carControl'].latActive = True                     # Carrot can steer without ACC: still "driving"
  sm['selfdriveState'].alertType = 'driverDistracted2/warning'
  sm['carState'].leftBlinker = True
  sm['carState'].steeringAngleDeg = 30.0                   # wheel turned left
  inp = face.f._inputs()
  assert inp.engaged and inp.dm_alert and inp.left_blinker
  assert inp.steer_deg == -30.0                            # the face looks and leans left
  sm['selfdriveState'].alertType = 'fcw/critical'
  assert face.f._inputs().fcw
  sm.recv_frame['carState'] = 9                          # a previous drive's car state is ignored
  inp = face.f._inputs()
  assert not inp.left_blinker and inp.v_ego == 0.0


def test_inputs_read_driving_events(face):
  sm = face.ui.sm
  inp = face.f._inputs()
  assert (inp.lane_phase, inp.lead_d, inp.cam_dist, inp.route_left) == (0, 0.0, 0.0, 0.0)
  assert not (inp.cutin or inp.gas or inp.blindspot_left) and inp.steer_torque == 0
  sm['modelV2'].meta = NS(laneChangeState='laneChangeStarting', laneChangeDirection='right')
  sm['radarState'].leadOne = NS(status=True, dRel=9.5, vLead=1.2)
  sm['controlsState'].curvature = 0.02
  sm['carrotMan'] = NS(xSpdDist=280, xSpdLimit=80, nGoPosDist=1500)
  sm['onroadEvents'] = [NS(name='steerSaturated')]
  sm['carState'].rightBlindspot = sm['carState'].gasPressed = True
  inp = face.f._inputs()
  assert (inp.lane_phase, inp.lane_dir) == (1, 1)
  assert (inp.lead_d, inp.lead_v, inp.cam_dist, inp.cam_limit, inp.route_left) == (9.5, 1.2, 280.0, 80.0, 1500.0)
  assert inp.lat_accel == pytest.approx(-0.02 * 10 * 10)        # curvature + is a right bend: lateral + is left
  assert not inp.cutin and inp.blindspot_right and inp.gas
  sm['modelV2'].meta = NS(laneChangeState='laneChangeFinishing', laneChangeDirection='left')
  sm['onroadEvents'] = []
  sm['selfdriveState'].alertType = 'radarCutin/warning'   # the one-frame event is gone; its alert is still up
  inp = face.f._inputs()
  assert (inp.lane_phase, inp.lane_dir) == (2, -1) and inp.cutin
  sm['selfdriveState'].alertType = 'belowSteerSpeed/warning'
  assert face.f._inputs().slow_steer
  sm.recv_frame['carrotMan'] = sm.recv_frame['radarState'] = 9   # left over from a previous drive
  inp = face.f._inputs()
  assert inp.cam_dist == 0 and inp.lead_d == 0


def test_steering_torque_reads_the_sent_torque_as_plus_right(face):
  sm = face.ui.sm
  sm['carOutput'].actuatorsOutput.torque = 0.8                # pushing the wheel left
  assert face.f._inputs().steer_torque == 0                   # only while it is steering
  sm['carControl'].latActive = True
  assert face.f._inputs().steer_torque == pytest.approx(-0.8)
  sm.alive['carOutput'] = False                               # no output: what the controller asked for
  sm['carControl'].actuators.torque = -0.6
  assert face.f._inputs().steer_torque == pytest.approx(0.6)


def test_straining_draws_gritted_teeth_and_shut_eyes(face):
  m = face.f.model
  m.effort, m.effort_side = 1.0, 1
  face.f.draw(face.rect)
  lines = [c for c in face.calls if c[0] == 'draw_line_ex']
  assert len(lines) >= 6 + 2 * 2                              # the zigzag mouth and a >< stroke per eye


def test_listen_feeds_the_microphone_and_dances_on_its_beat(face):
  import time
  import numpy as np
  module_time = time.monotonic
  rate, bpm = 16000, 120
  t = np.arange(int(12 * rate)) / rate
  kick = np.zeros_like(t)
  for b in np.arange(0, 12, 60 / bpm):
    i = int(b * rate)
    tt = t[:int(0.25 * rate)]
    kick[i:i + tt.size] += np.sin(2 * np.pi * (60 + 80 * np.exp(-tt * 12)) * tt)[:kick.size - i] * np.exp(-tt * 10)[:kick.size - i]
  audio = (kick * 12000).astype(np.int16)
  chunks = [audio[i:i + 800] for i in range(0, audio.size, 800)]
  feed = iter(chunks)
  face.f._audio_chunks = lambda: [(next(feed), module_time())]
  beats = 0
  for _ in range(len(chunks)):
    inp = face.f._inputs()
    face.f._listen(0.05, inp)
    beats += len(inp.music_beats)
  assert inp.music_on and abs(60 / inp.music_period - bpm) < 3 and beats >= 8
  face.ui.sm['selfdriveState'].alertSound = 'prompt'       # the car's own alert sound: not music
  face.f._audio_chunks = lambda: [(chunks[0], module_time())]
  inp = face.f._inputs()
  face.f._listen(0.05, inp)
  assert not inp.music_on


def test_event_props_draw(face):
  m = face.f.model
  m.spawn(kind='streak', x=-150, y=0, life=0.5, direction=-1)
  m.spawn(kind='cam', x=178, y=-86, life=1.8, age=0.4)
  face.f.draw(face.rect)
  n = names(face.calls)
  assert n.count('draw_circle_v') >= 2                    # the camera lens
  assert 'draw_line_ex' in n and 'draw_rectangle_rounded' in n


def test_render_draws_both_eyes_and_keeps_the_matrix_balanced(face):
  face.f.render(face.rect)                                # a new drive: the hello starts with the eyes shut
  for _ in range(60):
    face.f.model.update(0.05, face.f._inputs())
  face.calls.clear()
  face.f.render(face.rect)
  n = names(face.calls)
  assert n[0] == 'draw_rectangle_rec'
  assert n.count('rl_push_matrix') == n.count('rl_pop_matrix') == 1
  strips = [c for c in face.calls if c[0] == 'draw_triangle_strip']
  assert len(strips) >= 8                                 # both eye bodies, filled in gradient bands
  assert len({c[3] for c in strips}) >= 4                 # the bands step through colours
  assert n.count('rl_vertex2f') == 0                      # no per-vertex calls for the eyes
  layers = type(face.f).__init__.__globals__['GLOW_LAYERS']
  assert sum(1 for c in strips if c[3].a < 60) >= 2 * layers   # a halo around each eye, following its outline


@pytest.mark.parametrize('shape', ['heart', 'star'])
def test_special_eyes(face, shape):
  m = face.f.model
  if shape == 'heart':
    m.eye_shape = 'heart'
  else:
    m.stars(0.5)
  face.f.draw(face.rect)
  assert names(face.calls).count('rl_vertex2f') >= 60          # two filled stars or hearts


def test_shut_eyes_draw_a_stroke_and_props_draw(face):
  m = face.f.model
  m.P['up'].v = 1.0
  for k in ('p_shades', 'p_quest', 'p_bulb', 'p_nightcap'):
    m.P[k].v = 1.0
  m.spawn(kind='z', x=150, y=-40, life=2, size=14)
  m.spawn(kind='bang', x=196, y=-84, life=1.4)
  m.spawn(kind='sweat', x=-186, y=-64, life=2)
  m.spawn(kind='note', x=120, y=-50, life=1.6)
  m.spawn(kind='heart', x=0, y=-20, life=1.4, size=8)
  m.spawn(kind='spark', x=100, y=-40, life=1, size=6)
  m.spawn(kind='chev', x=206, y=-8, life=0.5)
  face.f.draw(face.rect)
  n = names(face.calls)
  assert n.count('draw_line_ex') > 10
  assert 'draw_rectangle_rounded' in n and 'draw_rectangle_rounded_lines_ex' in n
  texts = [c[2] for c in face.calls if c[0] == 'draw_text_ex']
  assert '?' in texts and 'Z' in texts
