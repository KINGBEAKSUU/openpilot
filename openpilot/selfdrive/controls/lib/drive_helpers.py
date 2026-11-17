import math

import numpy as np
from openpilot.cereal import log
from opendbc.car.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from opendbc.car.volkswagen.values import VolkswagenFlags


def is_volkswagen_meb(CP) -> bool:
  # VW MEB(ID.4/ID.5) 판별 단일 소스. 플랫폼 플래그 기반이라 향후 다른 VW angle 제어
  # 차종이 추가되어도 MEB 전용 경로(곡률 폐루프, 리드선택, FCW 게이트)가 오활성되지 않음.
  return CP.brand == "volkswagen" and bool(CP.flags & VolkswagenFlags.MEB)
import numpy as np

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MAX_VEL_ERR = 5.0  # m/s

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0  # m/s^2
MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED = 4.5  # m/s^2

def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

def get_lag_adjusted_curvature(CP, v_ego, psis, curvatures, steer_actuator_delay, distances):
  if len(psis) != CONTROL_N:
    psis = [0.0]*CONTROL_N
    curvatures = [0.0]*CONTROL_N
    distances = [0.0] * CONTROL_N
  v_ego = max(MIN_SPEED, v_ego)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  delay = max(0.01, steer_actuator_delay)

  # MPC can plan to turn the wheel and turn back before t_delay. This means
  # in high delay cases some corrections never even get commanded. So just use
  # psi to calculate a simple linearization of desired curvature
  current_curvature_desired = curvatures[0]
  delayed_curvature_desired = np.interp(delay, ModelConstants.T_IDXS[:CONTROL_N], curvatures)
  future_curvature_desired = np.interp(1.2, ModelConstants.T_IDXS[:CONTROL_N], curvatures)

  psi = np.interp(delay, ModelConstants.T_IDXS[:CONTROL_N], psis)

  distance = max(np.interp(delay, ModelConstants.T_IDXS[:CONTROL_N], distances), 0.001)
  #average_curvature_desired = psi / (v_ego * delay)

  # curve -> straight or reverse curve
  if v_ego > 5 and abs(current_curvature_desired) > 0.002 and \
     (abs(future_curvature_desired) < 0.001 or np.sign(current_curvature_desired) != np.sign(future_curvature_desired)):
    psis_damping = 0.2
  else:
    psis_damping = 1.0
  #psi *= psis_damping


  average_curvature_desired = psi / distance
  desired_curvature = 2 * average_curvature_desired - current_curvature_desired

  # This is the "desired rate of the setpoint" not an actual desired rate
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego**2) # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  safe_desired_curvature = np.clip(desired_curvature,
                                current_curvature_desired - max_curvature_rate * DT_MDL,
                                current_curvature_desired + max_curvature_rate * DT_MDL)
  return safe_desired_curvature

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val

def clip_curvature(v_ego, prev_curvature, new_curvature, roll,
                   max_lat_accel_low=MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED,
                   max_lat_accel_high=MAX_LATERAL_ACCEL_NO_ROLL,
                   max_lat_jerk=MAX_LATERAL_JERK):
  # This function respects ISO lateral jerk and acceleration limits + a max curvature.
  # The limits default to the EU-guideline constants; controlsd overrides them with
  # the LatAccelMaxLow/High and LatJerkMax params (ej tab, real-time adjustable).
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = max_lat_jerk / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)

  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  # Avoid an abrupt authority drop at 100 km/h. Gradually transition from
  # the low-speed allowance to the high-speed EU-guideline limit.
  max_lateral_accel_no_roll = float(np.interp(
    v_ego,
    [80 / 3.6, 120 / 3.6],
    [max_lat_accel_low, max_lat_accel_high],
  ))
  max_lat_accel = max_lateral_accel_no_roll + roll_compensation
  min_lat_accel = -max_lateral_accel_no_roll + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv


def read_lat_accel_limits(params):
  """clip_curvature에 쓸 횡가속/저크 한도를 파라미터에서 읽는다.

  단위: 파라미터 값은 0.1 m/s^2 (횡가속) / 0.1 m/s^3 (저크) 단위의 정수.
  예: LatAccelMaxLow=45 -> 4.5 m/s^2.
  - 값 파일이 없으면 params_keys.h에 등록된 기본값(저속 45 / 고속 30 / 저크 50)을 쓴다.
  - 키가 등록되지 않은 구 빌드(UnknownKeyName)나 읽기 오류 시 상수 기본값으로 폴백한다.
  - 0 이하나 비정상 값은 안전 범위로 클립한다. 한도가 0이 되면 곡률이 0으로 잘려
    조향이 안 되므로, 하한 0.5 m/s^2를 둔다.
  """
  try:
    low = params.get("LatAccelMaxLow", return_default=True)
    high = params.get("LatAccelMaxHigh", return_default=True)
    jerk = params.get("LatJerkMax", return_default=True)
    low = float(low) * 0.1 if low is not None else MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED
    high = float(high) * 0.1 if high is not None else MAX_LATERAL_ACCEL_NO_ROLL
    jerk = float(jerk) * 0.1 if jerk is not None else MAX_LATERAL_JERK
  except Exception:
    return (MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED, MAX_LATERAL_ACCEL_NO_ROLL, MAX_LATERAL_JERK)
  if not math.isfinite(low):
    low = MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED
  if not math.isfinite(high):
    high = MAX_LATERAL_ACCEL_NO_ROLL
  if not math.isfinite(jerk):
    jerk = MAX_LATERAL_JERK
  return (float(np.clip(low, 0.5, 10.0)),
          float(np.clip(high, 0.5, 10.0)),
          float(np.clip(jerk, 0.5, 20.0)))


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL, vEgoStopping=0.05):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
    v_target_1sec = np.interp(action_t + 1.0, t_idxs, speeds)
    v_max = np.max(speeds)
  else:
    v_now = 0.0
    a_now = 0.0
    v_target = 0.0
    v_target_1sec = 0.0
    a_target = 0.0
    v_max = 0.0
  should_stop = (v_target < vEgoStopping and
                 v_target_1sec < vEgoStopping)
  return a_target, should_stop, v_now, v_max #v_target #v_now

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
