import math
import time

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


# 최소거리 클램프의 레인리스 조향 보정 (ClampSteerCorrector)
# 레인리스 모드에서는 controlsd가 모델 곡률만으로 조향하므로, lateral planner의
# 경로 클램프가 조향에 닿지 않는다. 이 보정기는 controlsd에서 모델의 차선·도로경계
# 데이터를 직접 읽어 모델 곡률에 더할 곡률 보정을 계산한다. 경로 클램프와 같은
# 파라미터(ClampEnable/ClampMinDistLane/ClampMinDistEdge)와 같은 가드(확률
# 히스테리시스, 연석 std, 차선변경, 저속 페이드인)를 쓴다.
CLAMP_STEER_PROB_ON = 0.5
CLAMP_STEER_PROB_OFF = 0.3
CLAMP_STEER_EDGE_STD_MAX = 0.5
CLAMP_STEER_LOOKAHEAD_TIME = 1.0  # s, 룩어헤드 거리 = 속도 x 1초
CLAMP_STEER_LOOKAHEAD_MIN = 5.0  # m
CLAMP_STEER_LOOKAHEAD_MAX = 30.0  # m
CLAMP_STEER_MAX_LAT_ACCEL = 1.5  # m/s^2, 보정이 만드는 횡가속 상한
CLAMP_STEER_MAX_CURVATURE = 0.02  # 1/m, 보정 곡률 절대 상한 (저속 과조향 방지)
CLAMP_STEER_GAIN_DEFAULT = 1.0
CLAMP_STEER_GAIN_MAX = 2.0
CLAMP_STEER_MIN_DIST_LANE_DEFAULT = 1.35  # m, 경로 클램프와 동일
CLAMP_STEER_MIN_DIST_EDGE_DEFAULT = 1.15  # m
# 알림(띵) 조건: 보정 곡률 절댓값이 ON 이상이면 보정 중, OFF 아래로 떨어져야
# 종료 (히스테리시스). 종료 후 COOLDOWN 동안은 다시 켜지지 않아 띵 소리가
# 연속으로 울리지 않는다. 0.001 1/m은 60km/h에서 약 0.28 m/s²,
# 100km/h에서 약 0.77 m/s²의 횡가속에 해당 — 운전자가 느낄 만한 보정이다.
CLAMP_STEER_ALERT_ON_THRESHOLD = 0.001  # 1/m
CLAMP_STEER_ALERT_OFF_THRESHOLD = 0.0005  # 1/m
CLAMP_STEER_ALERT_COOLDOWN_SEC = 5.0  # s
# 차선변경 종료 후 홀드오프 + 보정 변화율 제한: 차선변경 중에는 보정이 꺼지고,
# 종료 직후 차가 새 차선에 안착하기 전에는 큰 위반이 남아 있을 수 있다.
# 홀드오프로 재개를 늦추고, 슬루 제한으로 재개 시 보정이 0에서 점진적으로
# 올라가게 해 급조향(확 틀어졌다 바로잡는 현상)을 막는다. 정상 주행에서는
# 위반이 서서히 커지므로 슬루 제한에 걸리지 않는다.
CLAMP_STEER_LANE_CHANGE_HOLDOFF_SEC = 2.0  # s
CLAMP_STEER_SLEW_RATE = 0.01  # 1/m/s, 최대 보정(0.0054)까지 약 0.54초
# 위반 불감대 (히스테리시스): 정상 주행에서 모델 경로가 선 쪽으로 조금씩
# 출렁이는 것만으로 보정이 자꾸 켜지는 것을 막는다. 위반이 20cm를 넘으면
# 켜지고, 10cm 아래로 떨어져야 꺼진다. 문제 지점(위반 약 60cm)에서는 그대로
# 동작하고, 보정 크기 자체는 줄이지 않는다(게이트만 건다).
CLAMP_STEER_DEADBAND_ON = 0.20  # m
CLAMP_STEER_DEADBAND_OFF = 0.10  # m


# 차선 중앙 유지 보정 (LaneCenterCorrector)
# 클램프가 "선을 넘지 말라"는 안전장치라면, 중앙 유지는 "차선 가운데에
# 있으라"는 지속 보정이다. 양쪽 차선 위치로 차선 중앙을 계산하고, 모델
# 경로가 중앙에서 벗어난 만큼 곡률 보정을 넣어 차를 가운데로 이끈다.
# 클램프와 같은 계산 방식(pure pursuit)과 같은 가드(확률 히스테리시스,
# 차선변경 홀드오프, 저속 페이드인, 슬루 제한)를 쓰지만, 양쪽 차선이
# 모두 보여야 동작한다 (중앙 계산에 양쪽이 필요). 클램프와 함께 동작한다:
# 중앙 유지가 평소에 차를 가운데 잡아주고, 클램프는 선에 가까워졌을 때
# 안전장치로 작동한다.
LANE_CENTER_PROB_ON = 0.5
LANE_CENTER_PROB_OFF = 0.3
LANE_CENTER_LOOKAHEAD_TIME = 1.0  # s, 룩어헤드 거리 = 속도 x 1초
LANE_CENTER_LOOKAHEAD_MIN = 5.0  # m
LANE_CENTER_LOOKAHEAD_MAX = 30.0  # m
LANE_CENTER_MAX_LAT_ACCEL = 1.5  # m/s^2, 보정이 만드는 횡가속 상한
LANE_CENTER_MAX_CURVATURE = 0.02  # 1/m, 보정 곡률 절대 상한 (저속 과조향 방지)
LANE_CENTER_GAIN_DEFAULT = 1.0
LANE_CENTER_GAIN_MAX = 2.0
LANE_CENTER_THRESHOLD_DEFAULT = 0.20  # m, 중앙에서 이만큼 벗어나면 보정 시작
LANE_CENTER_LANE_CHANGE_HOLDOFF_SEC = 2.0  # s
LANE_CENTER_SLEW_RATE = 0.01  # 1/m/s, 클램프와 동일


class ClampSteerCorrector:
  """레인리스 조향용 최소거리 클램프 보정기.

  update()는 모델 곡률에 더할 곡률 보정(1/m)을 반환한다.
  부호 규약 (controlsd와 동일): 양의 곡률 = 우회전(+y 방향). y는 +가 우측.
  왼쪽 선(중앙선)에 가까우면 양의 보정(우측으로 밀어냄), 오른쪽 선·연석에
  가까우면 음의 보정을 낸다. 위반이 없으면 0.0 (모델 조향을 그대로 둔다).

  모델 데이터가 불완전하거나 파라미터 읽기에 실패하면 0.0을 반환해
  controlsd가 죽지 않는다.
  """

  def __init__(self, params):
    self.params = params
    self.left_active = False
    self.right_active = False
    self.last_correction = 0.0
    # 알림(띵) 상태: 보정이 의미 있는 크기로 걸릴 때 True. controlsd가
    # controlsState.clampSteerActive로 selfdrived에 전달하고, selfdrived가
    # clampSteerActive 이벤트로 prompt 사운드를 울린다.
    self.alert_active = False
    self._alert_cooldown_until = 0.0
    # 차선변경 홀드오프 + 슬루 제한 상태. _prev_time이 None이거나 마지막 호출이
    # 0.5초보다 오래 전이면(보정기가 안 돌던 구간, 예: 레인풀 모드) 이전 출력을
    # 버리고 0에서 다시 시작한다.
    self._lane_change_holdoff_until = 0.0
    self._prev_kappa = 0.0
    self._prev_time = None
    # 불감대 게이트 상태: 위반이 20cm를 넘으면 True, 10cm 아래로 떨어지면 False.
    self._correction_active = False

  def _read_settings(self):
    """(enabled, min_dist_lane_m, min_dist_edge_m, gain)을 읽는다.

    get(..., return_default=True)을 쓴다. get_int/get_bool은 파일이 없으면
    0/False를 반환해서 fresh install에서 클램프가 꺼지거나 거리가 0이 되기
    때문이다. 키 미등록(구 빌드) 등 오류 시 상수 기본값으로 폴백한다.
    get()은 BOOL/INT를 제대로 변환해서 반환하므로 보통 bool/int가 온다.
    """
    try:
      enabled = self.params.get("ClampEnable", return_default=True)
      min_lane_cm = self.params.get("ClampMinDistLane", return_default=True)
      min_edge_cm = self.params.get("ClampMinDistEdge", return_default=True)
      gain_raw = self.params.get("ClampSteerGain", return_default=True)
    except Exception:
      return True, CLAMP_STEER_MIN_DIST_LANE_DEFAULT, CLAMP_STEER_MIN_DIST_EDGE_DEFAULT, CLAMP_STEER_GAIN_DEFAULT
    if enabled is None:
      enabled = True
    elif isinstance(enabled, str):
      enabled = enabled.strip().lower() in ("1", "true", "on", "yes")
    try:
      min_lane = (CLAMP_STEER_MIN_DIST_LANE_DEFAULT if min_lane_cm is None else float(
        np.clip(float(min_lane_cm) * 0.01, 0.0, 3.0)))
      min_edge = (CLAMP_STEER_MIN_DIST_EDGE_DEFAULT if min_edge_cm is None else float(
        np.clip(float(min_edge_cm) * 0.01, 0.0, 3.0)))
      gain = (CLAMP_STEER_GAIN_DEFAULT if gain_raw is None else float(
        np.clip(float(gain_raw) * 0.1, 0.0, CLAMP_STEER_GAIN_MAX)))
    except (TypeError, ValueError):
      return bool(enabled), CLAMP_STEER_MIN_DIST_LANE_DEFAULT, CLAMP_STEER_MIN_DIST_EDGE_DEFAULT, CLAMP_STEER_GAIN_DEFAULT
    if not math.isfinite(min_lane):
      min_lane = CLAMP_STEER_MIN_DIST_LANE_DEFAULT
    if not math.isfinite(min_edge):
      min_edge = CLAMP_STEER_MIN_DIST_EDGE_DEFAULT
    if not math.isfinite(gain):
      gain = CLAMP_STEER_GAIN_DEFAULT
    return bool(enabled), min_lane, min_edge, gain

  def _read_alert_enabled(self):
    """ClampSteerAlert 파라미터를 읽는다. 기본값 True (알림 켜짐).

    get(..., return_default=True)을 쓴다. 키 미등록(구 빌드) 등 오류 시
    True로 폴백해 알림이 계속 울린다.
    """
    try:
      v = self.params.get("ClampSteerAlert", return_default=True)
    except Exception:
      return True
    if v is None:
      return True
    if isinstance(v, str):
      return v.strip().lower() in ("1", "true", "on", "yes")
    return bool(v)

  def _update_alert(self, kappa):
    """보정 크기에 따라 알림 상태(alert_active)를 갱신한다.

    보정 절댓값이 ON 임계값 이상이면 켜고, OFF 임계값 아래로 떨어져야
    끈다 (히스테리시스). 한 번 꺼지면 쿨다운(5초) 동안 다시 켜지지 않아
    띵 소리가 연속으로 울리지 않는다. 알림 파라미터가 꺼져 있으면 항상
    False. 좌/우 방향과 무관하게 절댓값으로 판단한다.
    """
    if not self._read_alert_enabled():
      self.alert_active = False
      return
    now = time.monotonic()
    if self.alert_active:
      if abs(kappa) < CLAMP_STEER_ALERT_OFF_THRESHOLD:
        self.alert_active = False
        self._alert_cooldown_until = now + CLAMP_STEER_ALERT_COOLDOWN_SEC
    elif abs(kappa) >= CLAMP_STEER_ALERT_ON_THRESHOLD and now >= self._alert_cooldown_until:
      self.alert_active = True

  def update(self, md, v_ego, lane_changing):
    """모델 곡률에 더할 곡률 보정(1/m)을 계산한다. 양수 = 우회전.

    반환되는 보정 크기에 따라 알림 상태(alert_active)도 함께 갱신한다.
    보정이 0이면 알림은 꺼지는 방향으로 간다 (쿨다운은 _update_alert에서
    관리). 모든 early return 경로가 이 래퍼를 거치므로 알림 상태가
    stale로 남지 않는다.
    """
    kappa = self._update_impl(md, v_ego, lane_changing)
    kappa = self._slew_limit(kappa)
    self.last_correction = kappa
    self._update_alert(kappa)
    return kappa

  def _slew_limit(self, kappa):
    """보정 곡률의 변화율을 제한한다 (급조향 방지).

    목표 보정과 이전 출력의 차이를 CLAMP_STEER_SLEW_RATE * dt 이내로 자른다.
    차선변경 종료 직후처럼 큰 위반이 갑자기 나타나는 경우에도 보정이 0에서
    최대까지 약 0.5초에 걸쳐 점진적으로 올라간다. 정상 주행에서는 차가 선에
    서서히 다가가므로 목표 보정의 변화가 느려 제한에 걸리지 않는다.
    첫 호출이거나 0.5초 이상 호출이 없었으면(보정기가 안 돌던 구간) 이전
    출력을 버리고 0에서 시작한다. dt는 0.1초로 상한을 두어 긴 간격이
    제한을 무력화하지 못하게 한다.
    """
    now = time.monotonic()
    if self._prev_time is None or (now - self._prev_time) > 0.5:
      self._prev_time = now
      self._prev_kappa = 0.0
      return 0.0
    dt = min(max(now - self._prev_time, 0.0), 0.1)
    self._prev_time = now
    max_delta = CLAMP_STEER_SLEW_RATE * dt
    self._prev_kappa += float(np.clip(kappa - self._prev_kappa, -max_delta, max_delta))
    return self._prev_kappa

  def _update_impl(self, md, v_ego, lane_changing):
    """update()의 본체. 보정 곡률만 계산하고 알림 상태는 건드리지 않는다."""
    self.last_correction = 0.0
    enabled, min_lane, min_edge, gain = self._read_settings()
    if not enabled or gain <= 0.0:
      self.left_active = False
      self.right_active = False
      self._correction_active = False
      return 0.0
    # 차선변경 중에는 선을 넘는 것이 의도된 동작이므로 보정하지 않는다.
    # 차선변경이 끝난 뒤에도 홀드오프(2초) 동안은 보정하지 않는다. desire/
    # laneChangeState가 꺼지는 시점은 타이머 기준이라 차가 새 차선에 안착하기
    # 전일 수 있고, 그때 큰 위반이 남아 있으면 보정이 0에서 최대까지 단번에
    # 뛰며 차를 확 틀어버리기 때문이다. 홀드오프 타이머는 차선변경 중 매
    # 사이클 갱신되므로, 종료 시점부터 정확히 홀드오프 시간만큼 유지된다.
    now = time.monotonic()
    if lane_changing:
      self.left_active = False
      self.right_active = False
      self._correction_active = False
      self._lane_change_holdoff_until = now + CLAMP_STEER_LANE_CHANGE_HOLDOFF_SEC
      return 0.0
    if now < self._lane_change_holdoff_until:
      self._correction_active = False
      return 0.0
    # 저속 페이드인: 5km/h 이하 0, 10km/h 이상 100% (경로 클램프와 동일 구간)
    strength = float(np.interp(v_ego * 3.6, [5.0, 10.0], [0.0, 1.0]))
    if strength <= 0.0:
      self._correction_active = False
      return 0.0
    try:
      return self._compute(md, v_ego, min_lane, min_edge, gain, strength)
    except Exception:
      self._correction_active = False
      return 0.0

  def _compute(self, md, v_ego, min_lane, min_edge, gain, strength):
    lane_lines = md.laneLines
    edges = md.roadEdges
    lines_ok = (len(lane_lines) >= 4 and len(lane_lines[1].y) >= 2
                and len(lane_lines[2].y) >= 2 and len(md.laneLineProbs) >= 3)
    edges_ok = (len(edges) >= 2 and len(edges[0].y) >= 2 and len(edges[1].y) >= 2
                and len(md.roadEdgeStds) >= 2)
    pos_ok = (len(md.position.x) >= 2 and len(md.position.y) >= 2)
    if not pos_ok or (not lines_ok and not edges_ok):
      return 0.0

    # 확률 히스테리시스: 0.5를 넘으면 켜고 0.3 아래로 떨어져야 끈다 (경로 클램프와 동일).
    if lines_ok:
      l_prob = float(md.laneLineProbs[1])
      r_prob = float(md.laneLineProbs[2])
      if l_prob > CLAMP_STEER_PROB_ON:
        self.left_active = True
      elif l_prob < CLAMP_STEER_PROB_OFF:
        self.left_active = False
      if r_prob > CLAMP_STEER_PROB_ON:
        self.right_active = True
      elif r_prob < CLAMP_STEER_PROB_OFF:
        self.right_active = False
    else:
      self.left_active = False
      self.right_active = False

    # 룩어헤드 거리: 1초 앞 지점 (5~30m). 고속일수록 멀리 보고 미리 조향한다.
    lookahead = float(np.clip(v_ego * CLAMP_STEER_LOOKAHEAD_TIME,
                              CLAMP_STEER_LOOKAHEAD_MIN, CLAMP_STEER_LOOKAHEAD_MAX))
    path_y = float(np.interp(lookahead, md.position.x, md.position.y))
    if not math.isfinite(path_y):
      return 0.0

    # 좌측 하한: 경로는 이 값보다 우측(+y)이어야 한다. 차선이 보이면 차선 기준,
    # 안 보이면 도로경계 기준(연석 std가 작을 때만). 경로 클램프와 동일 로직.
    lo = -np.inf
    if self.left_active:
      left_y = float(np.interp(lookahead, lane_lines[1].x, lane_lines[1].y))
      lo = left_y + min_lane
    elif edges_ok:
      le_std = float(md.roadEdgeStds[0])
      if le_std < CLAMP_STEER_EDGE_STD_MAX:
        left_edge_y = float(np.interp(lookahead, edges[0].x, edges[0].y)) + le_std * 0.4
        lo = left_edge_y + min_edge
    # 우측 하한: 경로는 이 값보다 좌측(-y)이어야 한다.
    hi = np.inf
    if self.right_active:
      right_y = float(np.interp(lookahead, lane_lines[2].x, lane_lines[2].y))
      hi = right_y - min_lane
    elif edges_ok:
      re_std = float(md.roadEdgeStds[1])
      if re_std < CLAMP_STEER_EDGE_STD_MAX:
        right_edge_y = float(np.interp(lookahead, edges[1].x, edges[1].y)) - re_std * 0.4
        hi = right_edge_y - min_edge
    if not math.isfinite(lo):
      lo = -np.inf
    if not math.isfinite(hi):
      hi = np.inf
    if lo == -np.inf and hi == np.inf:
      return 0.0

    # 목표 y: 위반한 만큼만 밀어낸 위치. 양쪽 하한이 충돌하면(좁은 차로)
    # 중간 = 차선 중앙을 목표로 한다.
    if lo > hi:
      target_y = (lo + hi) / 2.0
    else:
      target_y = float(np.clip(path_y, lo, hi))
    # 위반 불감대 (히스테리시스): 위반이 20cm를 넘으면 켜지고 10cm 아래로
    # 떨어져야 꺼진다. 작은 위반은 무시해 정상 주행에서 보정이 자꾸 켜지는
    # 것을 막는다. 보정 크기는 줄이지 않는다(게이트만 건다). 켜질 때의
    # 목표값 점프는 슬루 제한이 부드럽게 올려준다.
    violation = abs(target_y - path_y)
    if self._correction_active:
      if violation < CLAMP_STEER_DEADBAND_OFF:
        self._correction_active = False
    elif violation > CLAMP_STEER_DEADBAND_ON:
      self._correction_active = True
    if not self._correction_active:
      return 0.0
    y_corr = (target_y - path_y) * strength
    if y_corr == 0.0:
      return 0.0

    # 곡률 변환 (pure pursuit): κ = 2·y / L². 양의 y_corr(우측 이동)는
    # 양의 곡률(우회전)이다. 게인을 곱하고, 보정이 만드는 횡가속이
    # 1.5 m/s²를 넘지 않도록, 그리고 곡률 자체가 0.02를 넘지 않도록
    # 자른다 (모델 곡률 위의 nudge이므로 과도한 개입을 막고, 최종 한도는
    # 뒤의 clip_curvature가 잡는다).
    kappa = 2.0 * y_corr / (lookahead ** 2) * gain
    max_kappa = min(CLAMP_STEER_MAX_LAT_ACCEL / max(v_ego, MIN_SPEED) ** 2,
                    CLAMP_STEER_MAX_CURVATURE)
    kappa = float(np.clip(kappa, -max_kappa, max_kappa))
    self.last_correction = kappa
    return kappa


class LaneCenterCorrector:
  """레인리스 조향용 차선 중앙 유지 보정기.

  update()는 모델 곡률에 더할 곡률 보정(1/m)을 반환한다.
  부호 규약 (controlsd와 동일): 양의 곡률 = 우회전(+y 방향). y는 +가 우측.
  모델 경로가 차선 중앙보다 왼쪽에 있으면 양의 보정(우측으로 이끔),
  오른쪽에 있으면 음의 보정을 낸다. 벗어남이 임계값 이하면 0.0 (모델
  조향을 그대로 둔다).

  클램프와 같은 계산 방식(pure pursuit)과 같은 가드(확률 히스테리시스,
  차선변경 홀드오프, 저속 페이드인, 슬루 제한)를 쓰지만, 양쪽 차선이
  모두 보여야 동작한다. 모델 데이터가 불완전하거나 파라미터 읽기에
  실패하면 0.0을 반환해 controlsd가 죽지 않는다.
  """

  def __init__(self, params):
    self.params = params
    self.left_active = False
    self.right_active = False
    self.last_correction = 0.0
    self._lane_change_holdoff_until = 0.0
    self._prev_kappa = 0.0
    self._prev_time = None
    # 불감대 게이트 상태: 벗어남이 임계값을 넘으면 True, 절반 아래로
    # 떨어지면 False.
    self._correction_active = False

  def _read_settings(self):
    """(enabled, threshold_m, gain)을 읽는다.

    get(..., return_default=True)을 쓴다. get_int/get_bool은 파일이 없으면
    0/False를 반환해서 fresh install에서 보정이 꺼지거나 임계값이 0이 되기
    때문이다. 키 미등록(구 빌드) 등 오류 시 상수 기본값으로 폴백한다.
    """
    try:
      enabled = self.params.get("LaneCenterEnable", return_default=True)
      threshold_cm = self.params.get("LaneCenterThreshold", return_default=True)
      gain_raw = self.params.get("LaneCenterGain", return_default=True)
    except Exception:
      return True, LANE_CENTER_THRESHOLD_DEFAULT, LANE_CENTER_GAIN_DEFAULT
    if enabled is None:
      enabled = True
    elif isinstance(enabled, str):
      enabled = enabled.strip().lower() in ("1", "true", "on", "yes")
    try:
      threshold = (LANE_CENTER_THRESHOLD_DEFAULT if threshold_cm is None else float(
        np.clip(float(threshold_cm) * 0.01, 0.0, 1.0)))
      gain = (LANE_CENTER_GAIN_DEFAULT if gain_raw is None else float(
        np.clip(float(gain_raw) * 0.1, 0.0, LANE_CENTER_GAIN_MAX)))
    except (TypeError, ValueError):
      return bool(enabled), LANE_CENTER_THRESHOLD_DEFAULT, LANE_CENTER_GAIN_DEFAULT
    if not math.isfinite(threshold):
      threshold = LANE_CENTER_THRESHOLD_DEFAULT
    if not math.isfinite(gain):
      gain = LANE_CENTER_GAIN_DEFAULT
    return bool(enabled), threshold, gain

  def update(self, md, v_ego, lane_changing):
    """모델 곡률에 더할 곡률 보정(1/m)을 계산한다. 양수 = 우회전."""
    kappa = self._update_impl(md, v_ego, lane_changing)
    kappa = self._slew_limit(kappa)
    self.last_correction = kappa
    return kappa

  def _slew_limit(self, kappa):
    """보정 곡률의 변화율을 제한한다 (급조향 방지). 클램프와 동일."""
    now = time.monotonic()
    if self._prev_time is None or (now - self._prev_time) > 0.5:
      self._prev_time = now
      self._prev_kappa = 0.0
      return 0.0
    dt = min(max(now - self._prev_time, 0.0), 0.1)
    self._prev_time = now
    max_delta = LANE_CENTER_SLEW_RATE * dt
    self._prev_kappa += float(np.clip(kappa - self._prev_kappa, -max_delta, max_delta))
    return self._prev_kappa

  def _update_impl(self, md, v_ego, lane_changing):
    """update()의 본체. 보정 곡률만 계산한다."""
    self.last_correction = 0.0
    enabled, threshold, gain = self._read_settings()
    if not enabled or gain <= 0.0:
      self.left_active = False
      self.right_active = False
      self._correction_active = False
      return 0.0
    # 차선변경 중에는 선을 넘는 것이 의도된 동작이므로 보정하지 않는다.
    # 종료 후 홀드오프(2초) 동안도 보정하지 않는다 (클램프와 동일 이유).
    now = time.monotonic()
    if lane_changing:
      self.left_active = False
      self.right_active = False
      self._correction_active = False
      self._lane_change_holdoff_until = now + LANE_CENTER_LANE_CHANGE_HOLDOFF_SEC
      return 0.0
    if now < self._lane_change_holdoff_until:
      self._correction_active = False
      return 0.0
    # 저속 페이드인: 5km/h 이하 0, 10km/h 이상 100% (클램프와 동일 구간)
    strength = float(np.interp(v_ego * 3.6, [5.0, 10.0], [0.0, 1.0]))
    if strength <= 0.0:
      self._correction_active = False
      return 0.0
    try:
      return self._compute(md, v_ego, threshold, gain, strength)
    except Exception:
      self._correction_active = False
      return 0.0

  def _compute(self, md, v_ego, threshold, gain, strength):
    lane_lines = md.laneLines
    lines_ok = (len(lane_lines) >= 4 and len(lane_lines[1].y) >= 2
                and len(lane_lines[2].y) >= 2 and len(md.laneLineProbs) >= 3)
    pos_ok = (len(md.position.x) >= 2 and len(md.position.y) >= 2)
    if not pos_ok or not lines_ok:
      return 0.0

    # 확률 히스테리시스: 0.5를 넘으면 켜고 0.3 아래로 떨어져야 끈다
    # (클램프와 동일). 양쪽 모두 켜져 있어야 중앙을 계산할 수 있다.
    l_prob = float(md.laneLineProbs[1])
    r_prob = float(md.laneLineProbs[2])
    if l_prob > LANE_CENTER_PROB_ON:
      self.left_active = True
    elif l_prob < LANE_CENTER_PROB_OFF:
      self.left_active = False
    if r_prob > LANE_CENTER_PROB_ON:
      self.right_active = True
    elif r_prob < LANE_CENTER_PROB_OFF:
      self.right_active = False
    if not (self.left_active and self.right_active):
      return 0.0

    # 룩어헤드 거리: 1초 앞 지점 (5~30m). 클램프와 동일.
    lookahead = float(np.clip(v_ego * LANE_CENTER_LOOKAHEAD_TIME,
                              LANE_CENTER_LOOKAHEAD_MIN, LANE_CENTER_LOOKAHEAD_MAX))
    path_y = float(np.interp(lookahead, md.position.x, md.position.y))
    left_y = float(np.interp(lookahead, lane_lines[1].x, lane_lines[1].y))
    right_y = float(np.interp(lookahead, lane_lines[2].x, lane_lines[2].y))
    if not (math.isfinite(path_y) and math.isfinite(left_y) and math.isfinite(right_y)):
      return 0.0

    # 차선 중앙: 양쪽 차선의 중간. 경로가 중앙에서 벗어난 만큼 보정한다.
    center_y = (left_y + right_y) / 2.0
    offset = path_y - center_y  # 양수면 경로가 중앙보다 우측
    # 불감대 (히스테리시스): 벗어남이 임계값을 넘으면 켜지고, 절반 아래로
    # 떨어져야 꺼진다. 작은 벗어남은 무시해 정상 주행에서 보정이 자꾸
    # 켜지는 것을 막는다. 보정 크기 자체는 줄이지 않는다(게이트만 건다).
    if self._correction_active:
      if abs(offset) < threshold / 2.0:
        self._correction_active = False
    elif abs(offset) > threshold:
      self._correction_active = True
    if not self._correction_active:
      return 0.0
    y_corr = (center_y - path_y) * strength
    if y_corr == 0.0:
      return 0.0

    # 곡률 변환 (pure pursuit): κ = 2·y / L². 양의 y_corr(우측 이동)는
    # 양의 곡률(우회전)이다. 게인을 곱하고, 보정이 만드는 횡가속이
    # 1.5 m/s²를 넘지 않도록, 그리고 곡률 자체가 0.02를 넘지 않도록
    # 자른다 (클램프와 동일 상한).
    kappa = 2.0 * y_corr / (lookahead ** 2) * gain
    max_kappa = min(LANE_CENTER_MAX_LAT_ACCEL / max(v_ego, MIN_SPEED) ** 2,
                    LANE_CENTER_MAX_CURVATURE)
    kappa = float(np.clip(kappa, -max_kappa, max_kappa))
    self.last_correction = kappa
    return kappa


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
