import math
import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
# from openpilot.common.swaglog import cloudlog
# from openpilot.common.logger import sLogger
from openpilot.common.params import Params

TRAJECTORY_SIZE = 33
# positive numbers go right
CAMERA_OFFSET = 0 #0.08
MIN_LANE_DISTANCE = 2.6
MAX_LANE_DISTANCE = 3.7
MAX_LANE_CENTERING_AWAY = 1.85
# 클램프 최소거리 기본값. 웹 ej 탭(ClampMinDistLane/ClampMinDistEdge)에서 바꾸면
# 그 값이 쓰이고, 이 상수는 Params를 읽을 수 없을 때의 폴백이다.
KEEP_MIN_DISTANCE_FROM_LANE = 1.35
KEEP_MIN_DISTANCE_FROM_EDGELANE = 1.15
# 최소거리 클램프(apply_min_distance_clamp)용 임계값
ROAD_EDGE_STD_MAX = 0.5  # 도로경계 신뢰도 상한: std가 이보다 크면 경계 위치를 믿지 않음
CLAMP_PROB_ON = 0.5      # 클램프 활성화 확률: 이 위면 해당 측 클램프 켜짐
CLAMP_PROB_OFF = 0.3     # 클램프 비활성화 확률: 이 아래로 떨어져야 꺼짐 (경계값 깜빡임 방지)

def clamp(num, min_value, max_value):
  # weird broken case, do something reasonable
  if min_value > num > max_value:
    return (min_value + max_value) * 0.5
  # ok, basic min/max below
  if num < min_value:
    return min_value
  if num > max_value:
    return max_value
  return num

def sigmoid(x, scale=1, offset=0):
  return (1 / (1 + math.exp(x*scale))) + offset

def lerp(start, end, t):
  t = clamp(t, 0.0, 1.0)
  return (start * (1.0 - t)) + (end * t)

def max_abs(a, b):
  return a if abs(a) > abs(b) else b

class LanePlanner:
  def __init__(self):
    self.ll_t = np.zeros((TRAJECTORY_SIZE,))
    self.ll_x = np.zeros((TRAJECTORY_SIZE,))
    self.lll_y = np.zeros((TRAJECTORY_SIZE,))
    self.rll_y = np.zeros((TRAJECTORY_SIZE,))
    self.le_y = np.zeros((TRAJECTORY_SIZE,))
    self.re_y = np.zeros((TRAJECTORY_SIZE,))
    #self.lane_width_estimate = FirstOrderFilter(3.2, 9.95, DT_MDL)
    self.lane_width_estimate = FirstOrderFilter(3.2, 3.0, DT_MDL)
    self.lane_width = 3.2
    self.lane_width_last = self.lane_width
    self.lane_change_multiplier = 1
    #self.lane_width_updated_count = 0

    self.lll_prob = 0.
    self.rll_prob = 0.
    self.d_prob = 0.

    self.lll_std = 0.
    self.rll_std = 0.

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.

    self.debugText = ""
    self.lane_width_left = 0.0
    self.lane_width_right = 0.0
    self.lane_width_left_filtered = FirstOrderFilter(1.0, 1.0, DT_MDL)
    self.lane_width_right_filtered = FirstOrderFilter(1.0, 1.0, DT_MDL)
    self.lane_offset_filtered = FirstOrderFilter(0.0, 2.0, DT_MDL)

    # 최소거리 클램프 상태
    self.l_prob = 0.
    self.r_prob = 0.
    self.edge_x = np.zeros((TRAJECTORY_SIZE,))
    self.le_std = float('inf')
    self.re_std = float('inf')
    self.edges_valid = False
    self.clamp_left_active = False
    self.clamp_right_active = False
    self.clamp_max_correction = 0.0
    self.clamp_sides = ""

    self.lanefull_mode = False
    self.d_prob_count = 0

    self.params = Params()

  def parse_model(self, md):

    lane_lines = md.laneLines
    edges = md.roadEdges

    if len(lane_lines) >= 4 and len(lane_lines[0].t) == TRAJECTORY_SIZE:
      self.ll_t = (np.array(lane_lines[1].t) + np.array(lane_lines[2].t))/2
      # left and right ll x is the same
      self.ll_x = lane_lines[1].x
      self.lll_y = np.array(lane_lines[1].y)
      self.rll_y = np.array(lane_lines[2].y)
      self.lll_prob = md.laneLineProbs[1]
      self.rll_prob = md.laneLineProbs[2]
      self.lll_std = md.laneLineStds[1]
      self.rll_std = md.laneLineStds[2]

    if len(edges[0].t) == TRAJECTORY_SIZE:
      self.le_y = np.array(edges[0].y) + md.roadEdgeStds[0] * 0.4
      self.re_y = np.array(edges[1].y) - md.roadEdgeStds[1] * 0.4
      self.edge_x = np.array(edges[0].x)
      self.le_std = float(md.roadEdgeStds[0])
      self.re_std = float(md.roadEdgeStds[1])
      self.edges_valid = True
    else:
      self.le_y = self.lll_y
      self.re_y = self.rll_y
      self.edges_valid = False

    desire_state = md.meta.desireState
    if len(desire_state):
      self.l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      self.r_lane_change_prob = desire_state[log.Desire.laneChangeRight]

  def get_d_path(self, CS, v_ego, path_t, path_xyz, curve_speed):
    #if v_ego > 0.1:
    #  self.lane_width_updated_count = max(0, self.lane_width_updated_count - 1)
    # Reduce reliance on lanelines that are too far apart or
    # will be in a few seconds
    l_prob, r_prob = self.lll_prob, self.rll_prob
    width_pts = self.rll_y - self.lll_y
    prob_mods = []
    for t_check in (0.0, 1.5, 3.0):
      width_at_t = np.interp(t_check * (v_ego + 7), self.ll_x, width_pts)
      #prob_mods.append(np.interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
      prob_mods.append(np.interp(width_at_t, [4.5, 6.0], [1.0, 0.0]))
    mod = min(prob_mods)
    l_prob *= mod
    r_prob *= mod

    # Reduce reliance on uncertain lanelines
    l_std_mod = np.interp(self.lll_std, [.15, .3], [1.0, 0.0])
    r_std_mod = np.interp(self.rll_std, [.15, .3], [1.0, 0.0])
    l_prob *= l_std_mod
    r_prob *= r_std_mod

    self.l_prob, self.r_prob = l_prob, r_prob

    # Find current lanewidth
    current_lane_width = abs(self.rll_y[0] - self.lll_y[0])

    max_updated_count = 10.0 * DT_MDL
    both_lane_available = False
    #speed_lane_width = np.interp(v_ego*3.6, [0., 60.], [2.8, 3.5])
    if l_prob > 0.5 and r_prob > 0.5 and self.lane_change_multiplier > 0.5:
      both_lane_available = True
      #self.lane_width_updated_count = max_updated_count
      self.lane_width_estimate.update(current_lane_width)
      self.lane_width_last = self.lane_width_estimate.x
    #elif self.lane_width_updated_count <= 0 and v_ego > 0.1:   # 양쪽차선이 없을때.... 일정시간후(10초)부터 speed차선폭 적용함.
    #  self.lane_width_estimate.update(speed_lane_width)
    else:
      self.lane_width_estimate.update(self.lane_width_last)

    self.lane_width =  self.lane_width_estimate.x
    clipped_lane_width = min(4.0, self.lane_width)
    path_from_left_lane = self.lll_y + clipped_lane_width / 2.0
    path_from_right_lane = self.rll_y - clipped_lane_width / 2.0

    # 가장 차선이 진한쪽으로 골라서..
    self.d_prob = max(l_prob, r_prob) if not both_lane_available else 1.0

    # 좌/우의 차선폭을 필터링.
    if self.lane_width_left > 0:
      self.lane_width_left_filtered.update(self.lane_width_left)
      #self.lane_width_left_filtered.x = self.lane_width_left #바로적용
    if self.lane_width_right > 0:
      self.lane_width_right_filtered.update(self.lane_width_right)
      #self.lane_width_right_filtered.x = self.lane_width_right #바로적용

    self.adjustLaneOffset = float(self.params.get_int("AdjustLaneOffset")) * 0.01
    self.adjustCurveOffset = self.adjustLaneOffset #float(self.params.get_int("AdjustCurveOffset")) * 0.01
    ADJUST_OFFSET_LIMIT = 0.4 #max(self.adjustLaneOffset, self.adjustCurveOffset)
    offset_curve = 0.0
    ## curve offset
    offset_curve = np.interp(abs(curve_speed), [50, 200], [self.adjustCurveOffset, 0.0]) * np.sign(curve_speed)

    offset_lane = 0.0
    if self.lane_width_left_filtered.x > 2.2 and self.lane_width_right_filtered.x > 2.2: #양쪽에 차로가 여유 있는경우
      offset_lane = 0.0
    elif self.lane_width_left_filtered.x < 2.0 and self.lane_width_right_filtered.x < 2.0: #양쪽에 차로가 여유 없는경우
      offset_lane = 0.0
    elif self.lane_width_left_filtered.x > self.lane_width_right_filtered.x:
      offset_lane = np.interp(self.lane_width, [2.5, 2.9], [0.0, self.adjustLaneOffset]) # 차선이 좁으면 안함..
    else:
      offset_lane = np.interp(self.lane_width, [2.5, 2.9], [0.0, -self.adjustLaneOffset]) # 차선이 좁으면 안함..

    #select lane path
    # 차선이 좁아지면, 도로경계쪽에 있는 차선 위주로 따라가도록함.
    if self.lane_width < 2.5:
      if r_prob > 0.5 and self.lane_width_right_filtered.x < self.lane_width_left_filtered.x:
        lane_path_y = path_from_right_lane
      elif l_prob > 0.5 and self.lane_width_left_filtered.x < 2.0:
        lane_path_y = path_from_left_lane
      else:
        lane_path_y = path_from_left_lane if l_prob > 0.5 or l_prob > r_prob else path_from_right_lane
    elif l_prob > 0.7 and r_prob > 0.7:
      lane_path_y = (path_from_left_lane + path_from_right_lane) / 2.
      # lane_width filtering에 의해서, 점점 줄어들때, 중앙선으로 붙어가는 현상이 생김.. 
      #if self.lane_width > 3.2:
      #  lane_path_y = path_from_right_lane
      #else:
      #  lane_path_y = (path_from_left_lane + path_from_right_lane) / 2.
    # 그외 진한차선을 따라가도록함.
    else:
      lane_path_y = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)

    use_laneless_center_adjust = False
    if use_laneless_center_adjust:
      ## 0.5초 앞의 중심을 보도록함.
      lane_path_y_center = np.interp(0.5, path_t, lane_path_y)
      path_xyz_y_center = np.interp(0.5, path_t, path_xyz[:,1])
      #lane_path_y_center = lane_path_y[0]
      #path_xyz_y_center = path_xyz[:,1][0]
      diff_center = (lane_path_y_center - path_xyz_y_center) if not self.lanefull_mode else 0.0
    else:
      diff_center = 0.0
    #print("center = {:.2f}={:.2f}-{:.2f}, lanefull={}".format(diff_center, lane_path_y_center, path_xyz_y_center, self.lanefull_mode))
    #diff_center = lane_path_y[5] - path_xyz[:,1][5] if not self.lanefull_mode else 0.0
    if offset_curve * offset_lane < 0:
      offset_total = np.clip(offset_curve + offset_lane + diff_center, - ADJUST_OFFSET_LIMIT, ADJUST_OFFSET_LIMIT)
    else:
      offset_total = np.clip(max(offset_curve, offset_lane, key=abs) + diff_center, - ADJUST_OFFSET_LIMIT, ADJUST_OFFSET_LIMIT)

    ## self.d_prob = 0 if lane_changing
    self.d_prob *= self.lane_change_multiplier  ## 차선변경중에는 꺼버림.
    if self.lane_change_multiplier < 0.5:
      #self.lane_offset_filtered.x = 0.0
      pass
    else:
      self.lane_offset_filtered.update(np.interp(self.d_prob, [0, 0.3], [0, offset_total]))

    ## laneless at lowspeed
    self.d_prob *= np.interp(v_ego*3.6, [5., 10.], [0.0, 1.0])

    #self.debugText = "OFFSET({:.2f}={:.2f}+{:.2f}+{:.2f}),Vc:{:.2f},dp:{:.1f},lf:{},lrw={:.1f}|{:.1f}|{:.1f}".format(
    #  self.lane_offset_filtered.x,
    #  diff_center, offset_lane, offset_curve,
    #  curve_speed,
    #  self.d_prob, self.lanefull_mode,
    #  self.lane_width_left_filtered.x, self.lane_width, self.lane_width_right_filtered.x)

    adjustLaneTime = self.params.get_float("LatMpcInputOffset") * 0.01 # 0.06 
    laneline_active = False
    self.d_prob_count = self.d_prob_count + 1 if self.d_prob > 0.3 else 0
    if self.lanefull_mode and self.d_prob_count > int(1 / DT_MDL):
      laneline_active = True
      use_dist_mode = False  ## 아무리생각해봐도.. 같은 방법인듯...
      if use_dist_mode:
        lane_path_y_interp = np.interp(path_xyz[:,0] + v_ego * adjustLaneTime, self.ll_x, lane_path_y)
        path_xyz[:,1] = self.d_prob * lane_path_y_interp + (1.0 - self.d_prob) * path_xyz[:,1]
      else:
        safe_idxs = np.isfinite(self.ll_t)
        if safe_idxs[0]:
          lane_path_y_interp = np.interp(path_t * (1.0 + adjustLaneTime), self.ll_t[safe_idxs], lane_path_y[safe_idxs])
          path_xyz[:,1] = self.d_prob * lane_path_y_interp + (1.0 - self.d_prob) * path_xyz[:,1]


    path_xyz[:, 1] += (CAMERA_OFFSET + self.lane_offset_filtered.x)

    self.offset_total = self.lane_offset_filtered.x

    return path_xyz, laneline_active

  def _read_clamp_settings(self):
    """클램프 설정을 Params에서 읽는다. (당근 웹 ej 탭에서 실시간 변경 가능)

    반환: (enabled, min_dist_lane_m, min_dist_edge_m)

    - get(..., return_default=True)을 쓴다. get_int/get_bool은 파일이 없으면
      0/False를 반환해서, 한 번도 저장하지 않은 fresh install에서 클램프가
      꺼지거나 거리가 0이 되기 때문이다. return_default=True면 파일이 없을 때
      params_keys.h에 등록된 기본값(켜짐, 135cm, 115cm)을 반환한다.
    - 키가 등록되지 않은 구 빌드에서 실행되면 UnknownKeyName이 나는데, 그 경우를
      포함해 읽기·변환에 실패하면 상수 기본값으로 폴백한다. 클램프가 꺼지거나
      controlsd가 죽지 않는다.
    - 거리는 웹에서 cm 단위 정수로 저장되므로 m로 변환하고 0~3m로 클립한다.
      (음수면 선 바깥쪽까지 허용하게 되므로 0 아래로는 내리지 않는다.)
    """
    try:
      enabled = self.params.get("ClampEnable", return_default=True)
      min_lane_cm = self.params.get("ClampMinDistLane", return_default=True)
      min_edge_cm = self.params.get("ClampMinDistEdge", return_default=True)
      if enabled is None:
        enabled = True
      elif isinstance(enabled, str):
        enabled = enabled.strip().lower() in ("1", "true", "on", "yes")
      if min_lane_cm is None:
        min_lane = KEEP_MIN_DISTANCE_FROM_LANE
      else:
        min_lane = float(np.clip(float(min_lane_cm) * 0.01, 0.0, 3.0))
      if min_edge_cm is None:
        min_edge = KEEP_MIN_DISTANCE_FROM_EDGELANE
      else:
        min_edge = float(np.clip(float(min_edge_cm) * 0.01, 0.0, 3.0))
      return bool(enabled), min_lane, min_edge
    except Exception:
      return True, KEEP_MIN_DISTANCE_FROM_LANE, KEEP_MIN_DISTANCE_FROM_EDGELANE

  def apply_min_distance_clamp(self, path_xyz, v_ego):
    """최종 경로가 감지된 차선·도로경계를 밟지 않도록 최소 거리를 유지한다.

    레인리스 모드에서는 차선 중심 보정이 꺼져 있어 모델 경로가 안쪽 선을 밟을 수
    있다. 이 클램프는 레인/레인리스 공통으로 동작하는 안전 가드이며, 경로가
    하한을 위반할 때만, 위반한 만큼만 밀어낸다. 평상시에는 경로에 손대지 않는다.

    - 좌/우 각각: 차선이 보이면 차선 기준, 차선이 안 보이면 도로경계 기준
      (경계 std가 작을 때만)으로 최소 거리를 유지한다. 거리 기본값은 차선
      1.35m·도로경계 1.15m이며 당근 웹 ej 탭에서 실시간으로 바꿀 수 있다.
    - 차선변경 중에는 차선을 넘는 것이 의도된 동작이므로 동작하지 않는다.
    - 저속에서는 횡단보도·주차선 등에 반응하지 않도록 서서히 활성화한다.
    - 차로가 너무 좁아 양쪽 하한이 충돌하면 차선 중앙으로 폴백한다.
    """
    self.clamp_max_correction = 0.0
    self.clamp_sides = ""
    enabled, min_dist_lane, min_dist_edge = self._read_clamp_settings()
    # 웹 ej 탭에서 끄면 클램프 전체를 끈다. 히스테리시스 상태도 초기화해서
    # 다시 켰을 때 깨끗하게 시작한다.
    if not enabled:
      self.clamp_left_active = False
      self.clamp_right_active = False
      return path_xyz
    # 차선변경 중이거나 ATC 동작 중에는 의도적으로 선을 넘으므로 클램프를 끈다.
    if self.lane_change_multiplier < 0.5:
      self.clamp_left_active = False
      self.clamp_right_active = False
      return path_xyz
    # 저속 페이드인: 5km/h 이하 0, 10km/h 이상 100% (d_prob 저속 페이드와 동일 구간)
    strength = float(np.interp(v_ego * 3.6, [5.0, 10.0], [0.0, 1.0]))
    if strength <= 0.0:
      return path_xyz

    # 히스테리시스: 0.5를 넘으면 켜고 0.3 아래로 떨어져야 끈다 (경계값 깜빡임 방지).
    if self.l_prob > CLAMP_PROB_ON:
      self.clamp_left_active = True
    elif self.l_prob < CLAMP_PROB_OFF:
      self.clamp_left_active = False
    if self.r_prob > CLAMP_PROB_ON:
      self.clamp_right_active = True
    elif self.r_prob < CLAMP_PROB_OFF:
      self.clamp_right_active = False

    n = path_xyz.shape[0]
    path_x = path_xyz[:, 0]
    lo = np.full(n, -np.inf)  # 좌측 하한: 경로는 이 값보다 우측이어야 한다
    hi = np.full(n, np.inf)   # 우측 하한: 경로는 이 값보다 좌측이어야 한다
    if self.clamp_left_active:
      left_y = np.interp(path_x, self.ll_x, self.lll_y)
      lo = np.maximum(lo, left_y + min_dist_lane)
    elif self.edges_valid and self.le_std < ROAD_EDGE_STD_MAX:
      # 좌측 차선이 안 보일 때만 도로경계로 폴백한다.
      left_edge_y = np.interp(path_x, self.edge_x, self.le_y)
      lo = np.maximum(lo, left_edge_y + min_dist_edge)
    if self.clamp_right_active:
      right_y = np.interp(path_x, self.ll_x, self.rll_y)
      hi = np.minimum(hi, right_y - min_dist_lane)
    elif self.edges_valid and self.re_std < ROAD_EDGE_STD_MAX:
      right_edge_y = np.interp(path_x, self.edge_x, self.re_y)
      hi = np.minimum(hi, right_edge_y - min_dist_edge)

    # 모델이 NaN을 뱉으면 해당 측 하한을 무효화한다 (NaN이 경로로 번지는 것 방지).
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isfinite(hi), hi, np.inf)
    if not np.isfinite(lo).any() and not np.isfinite(hi).any():
      return path_xyz

    target = np.clip(path_xyz[:, 1], lo, hi)
    # 양쪽 하한이 충돌하면(차로가 좁아 양쪽 최소거리 합보다 좁으면) 양쪽 경계의
    # 중간 = 차선 중앙으로 간다.
    conflict = lo > hi
    if np.any(conflict):
      target[conflict] = (lo[conflict] + hi[conflict]) / 2.0
    correction = target - path_xyz[:, 1]
    self.clamp_max_correction = float(np.max(np.abs(correction))) * strength
    if np.any(correction > 0.005):
      self.clamp_sides += "L"
    if np.any(correction < -0.005):
      self.clamp_sides += "R"
    path_xyz[:, 1] = path_xyz[:, 1] + strength * correction
    return path_xyz

  def calculate_plan_yaw_and_yaw_rate(self, path_xyz):
    if path_xyz.shape[0] < 3:
        # 너무 짧으면 직진 가정
        N = path_xyz.shape[0]
        return np.zeros(N), np.zeros(N)

    # x, y 추출
    x = path_xyz[:, 0]
    y = path_xyz[:, 1]

    # 모두 동일한 점인지 확인
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return np.zeros(len(x)), np.zeros(len(x))

    # 안전한 diff 계산
    dx = np.diff(x)
    dy = np.diff(y)
    mask = (dx == 0) & (dy == 0)
    dx[mask] = 1e-4
    dy[mask] = 0.0

    yaw = np.arctan2(dy, dx)
    yaw = np.append(yaw, yaw[-1])  # N-1 → N
    yaw = np.unwrap(yaw)

    dx_full = np.clip(np.diff(x), 1e-4, None)
    yaw_rate = np.diff(yaw) / dx_full
    yaw_rate = np.append(yaw_rate, yaw_rate[-1])
    yaw_rate = np.append(yaw_rate, 0.0)

    # NaN/Inf 방어
    if np.any(np.isnan(yaw_rate)) or np.any(np.isinf(yaw_rate)):
        yaw_rate = np.zeros_like(yaw_rate)
    if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
        yaw = np.zeros_like(yaw)

    return yaw, yaw_rate
