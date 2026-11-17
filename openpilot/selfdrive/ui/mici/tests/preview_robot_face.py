"""Offline native preview of the NewUi=3 robot face. Scripted inputs, no device or publishers."""
import argparse
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace as NS

import pyray as rl

# name: (seconds of script, what to set up)
CASES = {
  'idle': 3.0, 'engage-star': 0.35, 'engage-happy': 0.9, 'driving': 4.0, 'signal-right': 4.6, 'brake': 0.3,
  'driver-monitoring': 1.2, 'yawn': 50.2, 'asleep': 55.0, 'love': 1.0, 'cool': 1.2, 'think': 1.6, 'curious': 1.0,
  'straining': 2.2,
}
# driving-event reactions: case: (reaction, arg, engaged, seconds after it starts)
EVENTS = {
  'hello': ('hello', 0, False, 0.9), 'lane-go': ('lane_go', 1, True, 0.35), 'lane-done': ('lane_done', 0, True, 0.35),
  'blind-spot': ('bsm', 1, True, 0.4),
  'lead-go': ('lead_go', 0, False, 0.4), 'camera': ('camera', 0, True, 0.25), 'camera-fast': ('camera_fast', 0, True, 1.2),
  'slow-steer': ('slow_steer', 0, True, 0.6), 'steer-back': ('steer_back', 0, True, 0.3), 'cut-in': ('cutin', 0, True, 0.25), 'gas': ('gas', 0, True, 0.8),
  'arrive': ('arrive', 0, False, 0.9), 'curve': ('curve', 1, True, 0.4),
}
WARMUP = 2.6


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--case', choices=[*CASES, *EVENTS], action='append')
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)

  state_module = ModuleType('openpilot.selfdrive.ui.ui_state')
  state_module.ui_state = NS(started_frame=1, sm={})
  sys.modules[state_module.__name__] = state_module
  from openpilot.system.ui.lib.application import FontWeight, gui_app
  from openpilot.selfdrive.ui.mici.onroad.robot_face import RobotFace
  from openpilot.selfdrive.ui.mici.onroad.robot_face_model import FaceInputs, FaceModel

  gui_app.init_window('C4 robot face preview', font_weights=[FontWeight.BOLD])
  try:
    rect = rl.Rectangle(0, 0, 536, 240)
    for case in args.case or [*CASES, *EVENTS]:
      face = RobotFace()
      face.model = FaceModel(random.Random(5))
      inp = FaceInputs(session=1, v_ego=12.0)
      if case in EVENTS:
        reaction, arg, inp.engaged, after = EVENTS[case]
        t = 0.0
        while t < WARMUP:
          face.model.update(0.05, inp)
          t = round(t + 0.05, 4)
        face.model.react(reaction, arg)
        t = 0.0
        while t < after:
          face.model.update(0.05, inp)
          t = round(t + 0.05, 4)
        total = -1.0
      else:
        total = CASES[case]
      if case == 'driving':
        inp.engaged = True
      t = 0.0
      while t < total:
        if case.startswith('engage'):
          inp.engaged = True
        elif case == 'signal-right':
          inp.engaged, inp.right_blinker, inp.steer_deg = True, t > 3.5, 12.0
        elif case == 'straining':
          inp.engaged, inp.steer_torque = True, min(1.0, 0.4 + t * 0.4)
        elif case == 'brake':
          inp.a_ego = -4.5
        elif case == 'driver-monitoring':
          inp.dm_alert = True
        elif case in ('yawn', 'asleep'):
          inp.v_ego, inp.standstill = 0.0, True
        elif case in ('love', 'cool', 'think', 'curious') and abs(t - 0.05) < 1e-6:
          face.model.run_behavior(case)
        face.model.update(0.05, inp)
        t = round(t + 0.05, 4)
      rl.begin_drawing()
      rl.clear_background(rl.BLACK)
      face.draw(rect)
      rl.rl_draw_render_batch_active()
      image = rl.load_image_from_screen()
      try:
        if (image.width, image.height) != (536, 240):
          rl.image_resize(image, 536, 240)
        assert rl.export_image(image, str(args.output / f'{case}.png'))
      finally:
        rl.unload_image(image)
      rl.end_drawing()
    print(f'PASS: {len(args.case or [*CASES, *EVENTS])} native 536x240 robot frames; scripted inputs, not a vehicle capture')
  finally:
    gui_app.close()


if __name__ == '__main__':
  main()
