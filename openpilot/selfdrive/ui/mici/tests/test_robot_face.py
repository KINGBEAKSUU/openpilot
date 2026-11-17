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
      carState=NS(leftBlinker=False, rightBlinker=False, steeringAngleDeg=0.0, vEgo=10.0, aEgo=0.0, standstill=False),
      selfdriveState=NS(enabled=False, alertType=''), carControl=NS(latActive=False))
    names = ('carState', 'selfdriveState', 'carControl')
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
  inp = face.f._inputs()
  assert inp.engaged and inp.dm_alert and inp.left_blinker
  sm['selfdriveState'].alertType = 'fcw/critical'
  assert face.f._inputs().fcw
  sm.recv_frame['carState'] = 9                          # a previous drive's car state is ignored
  inp = face.f._inputs()
  assert not inp.left_blinker and inp.v_ego == 0.0


def test_render_draws_both_eyes_and_keeps_the_matrix_balanced(face):
  face.f.render(face.rect)
  n = names(face.calls)
  assert n[0] == 'draw_rectangle_rec'
  assert n.count('rl_push_matrix') == n.count('rl_pop_matrix') == 1
  strips = [c for c in face.calls if c[0] == 'draw_triangle_strip']
  assert len(strips) >= 8                                 # both eye bodies, filled in gradient bands
  assert len({c[3] for c in strips}) >= 4                 # the bands step through colours
  assert n.count('rl_vertex2f') == 0                      # no per-vertex calls for the eyes
  assert n.count('draw_texture_pro') >= 2                 # glow behind each eye


@pytest.mark.parametrize('shape', ['heart', 'star'])
def test_special_eyes(face, shape):
  m = face.f.model
  if shape == 'heart':
    m.eye_shape = 'heart'
  else:
    m.star_until = m.clock + 0.5
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
