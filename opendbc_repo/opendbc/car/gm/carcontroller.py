from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter

import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import Bus, DT_CTRL, apply_driver_steer_torque_limits, structs, create_gas_interceptor_command
from opendbc.car.gm import gmcan
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.gm.values import DBC, CanBus, CarControllerParams, CruiseButtons, GMFlags, EV_CAR, AccState, CAR, SDGM_CAR, ALT_ACCS
from opendbc.car.interfaces import CarControllerBase
from openpilot.selfdrive.controls.lib.drive_helpers import apply_deadzone
from opendbc.car.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

VisualAlert = structs.CarControl.HUDControl.VisualAlert
NetworkLocation = structs.CarParams.NetworkLocation
LongCtrlState = structs.CarControl.Actuators.LongControlState

# Camera cancels up to 0.1s after brake is pressed, ECM allows 0.5s
CAMERA_CANCEL_DELAY_FRAMES = 10
# Enforce a minimum interval between steering messages to avoid a fault
MIN_STEER_MSG_INTERVAL_MS = 15

# Constants for pitch compensation
PITCH_DEADZONE = 0.01  # [radians] 0.01 ≈ 1% grade
BRAKE_PITCH_FACTOR_BP = [5., 10.]  # [m/s] smoothly revert to planned accel at low speeds
BRAKE_PITCH_FACTOR_V = [0., 1.]  # [unitless in [0,1]]; don't touch

class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.start_time = 0.
    self.apply_torque_last = 0
    self.apply_gas = 0
    self.apply_brake = 0
    # kans: button spam
    self.apply_speed = 0
    self.frame = 0
    self.last_steer_frame = 0
    self.last_button_frame = 0
    self.cancel_counter = 0
    self.pedal_steady = 0.

    self.lka_steering_cmd_counter = 0
    self.lka_icon_status_last = (False, False)

    self.params = CarControllerParams(self.CP)
    self.params_ = Params() # kans: button spam

    self.packer_pt = CANPacker(DBC[self.CP.carFingerprint][Bus.pt])
    self.packer_obj = CANPacker(DBC[self.CP.carFingerprint][Bus.radar])
    self.packer_ch = CANPacker(DBC[self.CP.carFingerprint][Bus.chassis])

    self.long_pitch = False
    #self.use_ev_tables = False

    self.pitch = FirstOrderFilter(0., 0.09 * 4, DT_CTRL * 4)  # runs at 25 Hz
    self.accel_g = 0.0

    # Kans: AutoResume
    self.activateCruise_after_brake = False
    self.autoCruise_activate = False
    self.autoCruise_frame = 0
    self.resume_activate = False
    self.resume_frame = 0
    self._pending_activateCruise = False
    self.btn_rc_pt = -1
    self.btn_rc_cam = -1
    self._last_brake_idx = None  # 직전전송 idx 저장용
    self._brk_rc = -1
    self.cruiseDelay_time = 0.0
    self.resumeDelay_time = 0.0
    self._hill_detected = False
    self.accel_force = 0
    self.pulse_frame = 0
    self.resume_fault_guard = 0
    self.autoCruise_try_count = 0  # 오토크루즈 버튼 재시도 횟수
    self.last_pulse_reset_frame = 0  # 리쥼펄스 최소유지"보장"용

  def update(self, CC, CS, now_nanos):
    params = Params()
    if self.frame % 50 == 0:
      steerMax = params.get_int("CustomSteerMax")
      steerDeltaUp = params.get_int("CustomSteerDeltaUp")
      steerDeltaDown = params.get_int("CustomSteerDeltaDown")
      if steerMax > 0:
        self.params.STEER_MAX = steerMax
      if steerDeltaUp > 0:
        self.params.STEER_DELTA_UP = steerDeltaUp
      if steerDeltaDown > 0:
        self.params.STEER_DELTA_DOWN = steerDeltaDown

    self.long_pitch = params.get_bool("LongPitch")
    #self.use_ev_tables = params.get_bool("EVTable")

    actuators = CC.actuators
    accel = brake_accel = actuators.accel
    hud_control = CC.hudControl
    hud_alert = hud_control.visualAlert
    hud_v_cruise = hud_control.setSpeed
    if hud_v_cruise > 70:
      hud_v_cruise = 0

    # Send CAN commands.
    can_sends = []

    # Steering (Active: 50Hz, inactive: 10Hz)
    steer_step = self.params.STEER_STEP if CC.latActive else self.params.INACTIVE_STEER_STEP

    if self.CP.networkLocation == NetworkLocation.fwdCamera:
      # Also send at 50Hz:
      # - on startup, first few msgs are blocked
      # - until we're in sync with camera so counters align when relay closes, preventing a fault.
      #   openpilot can subtly drift, so this is activated throughout a drive to stay synced
      out_of_sync = self.lka_steering_cmd_counter % 4 != (CS.cam_lka_steering_cmd_counter + 1) % 4
      if CS.loopback_lka_steering_cmd_ts_nanos == 0 or out_of_sync:
        steer_step = self.params.STEER_STEP

    self.lka_steering_cmd_counter += 1 if CS.loopback_lka_steering_cmd_updated else 0

    # Avoid GM EPS faults when transmitting messages too close together: skip this transmit if we
    # received the ASCMLKASteeringCmd loopback confirmation too recently
    last_lka_steer_msg_ms = (now_nanos - CS.loopback_lka_steering_cmd_ts_nanos) * 1e-6
    if (self.frame - self.last_steer_frame) >= steer_step and last_lka_steer_msg_ms > MIN_STEER_MSG_INTERVAL_MS:
      # Initialize ASCMLKASteeringCmd counter using the camera until we get a msg on the bus
      if CS.loopback_lka_steering_cmd_ts_nanos == 0:
        self.lka_steering_cmd_counter = CS.pt_lka_steering_cmd_counter + 1

      if CC.latActive:
        new_torque = int(round(actuators.torque * self.params.STEER_MAX))
        apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.params)
      else:
        apply_torque = 0

      self.last_steer_frame = self.frame
      self.apply_torque_last = apply_torque
      idx = self.lka_steering_cmd_counter % 4
      can_sends.append(gmcan.create_steering_control(self.packer_pt, CanBus.POWERTRAIN, apply_torque, idx, CC.latActive))

    if self.CP.openpilotLongitudinalControl:
      # Gas/regen, brakes, and UI commands - all at 25Hz
      if self.frame % 4 == 0:
        friction_sent_this_tick = False
        self.cruiseDelay_time = params.get_float("CruiseDelay") * 0.01
        self.resumeDelay_time = params.get_float("ResumeDelay") * 0.01
        auto_cruise_enabled = params.get_int("AutoCruiseControl") > 0
        auto_engage_enabled = params.get_int("AutoEngage") == 2
        self.accel_force = params.get_int("AccelForce")

        # GM: softHold
        stopping = actuators.longControlState == LongCtrlState.stopping or CS.out.softHoldActive > 0

        # Pitch compensated acceleration;
        # TODO: include future pitch (sm['modelDataV2'].orientation.y) to account for long actuator delay
        if self.long_pitch and len(CC.orientationNED) > 1:
          self.pitch.update(CC.orientationNED[1])
          self.accel_g = ACCELERATION_DUE_TO_GRAVITY * apply_deadzone(self.pitch.x, PITCH_DEADZONE) # driving uphill is positive pitch
          accel += self.accel_g
          brake_accel = actuators.accel + self.accel_g * np.interp(CS.out.vEgo, BRAKE_PITCH_FACTOR_BP, BRAKE_PITCH_FACTOR_V)

        at_full_stop = CC.longActive and CS.out.standstill
        near_stop = CC.longActive and (abs(CS.out.vEgo) < self.params.NEAR_STOP_BRAKE_PHASE)
        interceptor_gas_cmd = 0

        # 언덕감지(accel_g가 클수록 높은 경사)
        if self.accel_g > 0.25:
          self._hill_detected = True
        else:
          self._hill_detected = False

        if not CC.longActive:
          # ASCM sends max regen when not enabled
          self.apply_gas = self.params.INACTIVE_REGEN
          self.apply_brake = 0
        elif near_stop and stopping and not CC.cruiseControl.resume:
          self.apply_gas = self.params.INACTIVE_REGEN
          self.apply_brake = int(min(-100 * self.CP.stopAccel, self.params.MAX_BRAKE))

        else:
          # Normal operation
          if self.CP.carFingerprint in EV_CAR:
            self.params.update_ev_gas_brake_threshold(CS.out.vEgo)
            self.apply_gas = int(round(np.interp(accel if self.long_pitch else actuators.accel, self.params.EV_GAS_LOOKUP_BP, self.params.GAS_LOOKUP_V)))
            self.apply_brake = int(round(np.interp(brake_accel if self.long_pitch else actuators.accel, self.params.EV_BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)))
          else:
            self.apply_gas = int(round(np.interp(accel if self.long_pitch else actuators.accel, self.params.GAS_LOOKUP_BP, self.params.GAS_LOOKUP_V)))
            self.apply_brake = int(round(np.interp(brake_accel if self.long_pitch else actuators.accel, self.params.BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)))

          # Don't allow any gas above inactive regen while stopping
          # FIXME: brakes aren't applied immediately when enabling at a stop
          if stopping:
            self.apply_gas = self.params.INACTIVE_REGEN

        if self.CP.enableGasInterceptorDEPRECATED and self.apply_gas > self.params.INACTIVE_REGEN and CS.out.cruiseState.standstill:
          # "Tap" the accelerator pedal to re-engage ACC
          interceptor_gas_cmd = self.params.SNG_INTERCEPTOR_GAS
          self.apply_brake = 0
          self.apply_gas = self.params.INACTIVE_REGEN

        idx = (self.frame // 4) % 4

        if self.CP.enableGasInterceptorDEPRECATED:
          can_sends.append(create_gas_interceptor_command(self.packer_pt, interceptor_gas_cmd, idx))
        if self.CP.carFingerprint not in ALT_ACCS:
          at_full_stop = CC.longActive and CS.out.standstill
          near_stop = CC.longActive and (abs(CS.out.vEgo) < self.params.NEAR_STOP_BRAKE_PHASE)
          friction_brake_bus = CanBus.CHASSIS
          # GM Camera exceptions
          # TODO: can we always check the longControlState?
          if self.CP.networkLocation == NetworkLocation.fwdCamera:
            at_full_stop = at_full_stop and stopping
            friction_brake_bus = CanBus.POWERTRAIN
            if self.CP.carFingerprint in SDGM_CAR:
              friction_brake_bus = CanBus.CHASSIS

          if self.CP.autoResumeSng:
            resume = actuators.longControlState != LongCtrlState.starting or CC.cruiseControl.resume
            at_full_stop = at_full_stop and not resume

          if CC.cruiseControl.resume and CS.pcm_acc_status == AccState.STANDSTILL:
            if self.CP.carFingerprint in EV_CAR:
              acc_engaged = False
            else:
              acc_engaged = CC.enabled
          else:
            acc_engaged = CC.enabled

          # Kans: 오토크루즈 대기 플래그
          if CS.out.activateCruise > 0:
            self._pending_activateCruise = True

          # Kans: 리쥼용 가속펄스 로직
          has_lead = (CS.lead_distance is not None) and np.isfinite(CS.lead_distance)
          d_rel = CS.lead_distance if has_lead else float('inf')
          lead_follow_ok = has_lead and (2.0 < CS.lead_distance < 10.0) and (CS.lead_speed > 0.1)
          # 신호 대기 상황: 앞차 없고, 스탠드스틸일 때
          no_lead_signal_ok = (not has_lead) and CS.out.standstill
          lead_ok = lead_follow_ok or no_lead_signal_ok

          # pid로그용 상태 문자열
          state_map = {LongCtrlState.off: "off", LongCtrlState.stopping: "stopping",
                       LongCtrlState.starting: "starting", LongCtrlState.pid: "pid"}
          state_name = state_map.get(actuators.longControlState, str(actuators.longControlState))

          resume_pulse = 0
          auto_longcontrol = auto_cruise_enabled or auto_engage_enabled
          acc_state = CS.out.accStatus  # = AcceleratorPedal2.CruiseState raw 값 (0~4)
          acc_not_engaged = acc_state in (AccState.OFF, AccState.STANDBY)

          # 펄스 관련 상수(언덕/평지 최소 유지시간 + 쿨다운)
          MIN_HILL_PULSE = 0.30   # 언덕에서 최소 0.30s는 밀어주기
          MIN_FLAT_PULSE = 0.20   # 평지에서 최소 0.20s는 밀어주기
          PULSE_COOLDOWN = 0.50   # 펄스 끊고 난 뒤 재시작까지 최소 0.50s 대기

          if self._hill_detected and auto_longcontrol:
            if actuators.longControlState == LongCtrlState.stopping:
              # 언덕 펄스 도중 정지 상태로 돌아가면 펄스 취소
              if self.pulse_frame != 0:
                dt = (self.frame - self.pulse_frame) * DT_CTRL
                print(f"[언덕펄스취소] 시간={dt:.2f}s 이유=stopping 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f} 거리={d_rel:.1f}")
              self.pulse_frame = 0
              self.last_pulse_reset_frame = self.frame
              resume_pulse = 0
            elif actuators.longControlState == LongCtrlState.starting:
              # 언덕 펄스 시작: 마지막 리셋 후 쿨다운이 지난 경우에만 시작
              if self.pulse_frame == 0 and lead_ok:
                cooldown_ok = (self.frame - self.last_pulse_reset_frame) * DT_CTRL > PULSE_COOLDOWN
                if cooldown_ok:
                  self.pulse_frame = self.frame
                  print(f"[언덕펄스시작] 차속도={CS.out.vEgo:.2f}m/s 액셀={actuators.accel:.1f} 각도={self.accel_g:.1f} 거리={d_rel:.1f}")
              # 펄스가 활성 상태이면 유지/종료 판단
              if self.pulse_frame != 0:
                dt = (self.frame - self.pulse_frame) * DT_CTRL
                if dt < 1.0:  # 언덕 펄스 유지시간 (최대 1.0s)
                  resume_pulse = int(self.accel_force)
                  at_full_stop = False
                else:
                  if 1.0 <= dt < 1.0 + DT_CTRL:  # 1.0s를 막 넘긴 시점에 한 번만 로그
                    print(f"[언덕펄스종료] 시간={dt:.2f}s 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f} 거리={d_rel:.1f}")
                  resume_pulse = 0
                  self.pulse_frame = 0
                  self.last_pulse_reset_frame = self.frame

            else:
              # 언덕인데 starting/lead_ok 조건을 벗어나면 펄스 취소
              if self.pulse_frame != 0:
                dt = (self.frame - self.pulse_frame) * DT_CTRL
                # 언덕에서는 최소 MIN_HILL_PULSE 만큼은 유지하려고 시도
                if dt < MIN_HILL_PULSE and CC.longActive and not CS.out.brakePressed:
                  # 너무 짧게 끊으려는 상황이면 한 번은 더 밀어줌
                  resume_pulse = int(self.accel_force)
                  at_full_stop = False
                  # 필요시 디버그용:
                  # print(f"[언덕펄스 유지] 시간={dt:.2f}s pid={state_name} 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f}")
                else:
                  print(f"[언덕펄스취소] 시간={dt:.2f}s pid={state_name} 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f} 거리={d_rel:.1f}")
                  resume_pulse = 0
                  self.pulse_frame = 0
                  self.last_pulse_reset_frame = self.frame
              else:
                resume_pulse = 0

          elif not self._hill_detected and at_full_stop and auto_longcontrol:
            if actuators.longControlState == LongCtrlState.stopping:
              # 평지 펄스 도중 다시 stopping으로 가면 취소
              if self.pulse_frame != 0:
                dt = (self.frame - self.pulse_frame) * DT_CTRL
                print(f"[평지펄스취소] 시간={dt:.2f}s 원인={state_name} 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f}")
              self.pulse_frame = 0
              self.last_pulse_reset_frame = self.frame
              resume_pulse = 0

            elif actuators.longControlState == LongCtrlState.starting:
              # 평지 펄스 시작: 마지막 리셋 후 쿨다운이 지난 경우에만 시작
              if self.pulse_frame == 0 and lead_ok:
                cooldown_ok = (self.frame - self.last_pulse_reset_frame) * DT_CTRL > PULSE_COOLDOWN
                if cooldown_ok:
                  self.pulse_frame = self.frame
                  print(f"[평지펄스시작] 차속도={CS.out.vEgo:.1f}m/s 액셀={actuators.accel:.1f} 각도={self.accel_g:.1f} 거리={d_rel:.1f} 리드속도={CS.lead_speed:.1f}")

              # 펄스가 활성 상태이면 유지/종료 판단
              if self.pulse_frame != 0:
                dt = (self.frame - self.pulse_frame) * DT_CTRL
                if dt < 1.0:  # 평지 펄스 유지시간 (최대 1.0s)
                  resume_pulse = int(self.accel_force)
                  at_full_stop = False
                else:
                  # 1.0s를 막 넘긴 시점에 한 번만 로그
                  if 1.0 <= dt < 1.0 + DT_CTRL:
                    print(f"[평지펄스종료] 시간={dt:.2f}s 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f} 거리={d_rel:.1f} 리드속도={CS.lead_speed:.1f}")
                  resume_pulse = 0
                  self.pulse_frame = 0
                  self.last_pulse_reset_frame = self.frame

            else:
              # 평지인데 starting/lead_ok 조건을 벗어나면 펄스 취소
              if self.pulse_frame != 0:
                dt = (self.frame - self.pulse_frame) * DT_CTRL
                # 평지에서도 최소 MIN_FLAT_PULSE 만큼은 유지
                if dt < MIN_FLAT_PULSE and CC.longActive and not CS.out.brakePressed:
                  resume_pulse = int(self.accel_force)
                  at_full_stop = False
                  # print(f"[평지펄스 유지] 시간={dt:.2f}s pid={state_name} 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f} 리드속도={CS.lead_speed:.1f}")
                else:
                  print(f"[평지펄스취소] 시간={dt:.2f}s pid={state_name} 차속도={CS.out.vEgo:.1f}m/s 각도={self.accel_g:.1f} 거리={d_rel:.1f} 리드속도={CS.lead_speed:.1f}")
                  resume_pulse = 0
                  self.pulse_frame = 0
                  self.last_pulse_reset_frame = self.frame
              else:
                resume_pulse = 0

          else:
            # 언덕도 아니고(at_full_stop 조건 밖) 또는 기타 상황:
            # -> 펄스/타이머 초기화
            if self.pulse_frame != 0:
              dt = (self.frame - self.pulse_frame) * DT_CTRL
              # 너무 짧게 끊으려는 경우에는 한 번 더 유지 (언덕/평지 공통 최소 시간 적용)
              min_pulse = MIN_HILL_PULSE if self._hill_detected else MIN_FLAT_PULSE
              if dt < min_pulse and CC.longActive and not CS.out.brakePressed:
                resume_pulse = int(self.accel_force)
                at_full_stop = False
                # print(f"[펄스 유지] 시간={dt:.2f}s 이유=언덕모드={self._hill_detected} 상태={at_full_stop} 각도={self.accel_g:.1f}")
              else:
                print(f"[펄스초기화] 시간={dt:.2f}s 이유=언덕모드={self._hill_detected} 상태={at_full_stop} 각도={self.accel_g:.1f}")
                self.pulse_frame = 0
                self.last_pulse_reset_frame = self.frame
                resume_pulse = 0
            else:
              resume_pulse = 0

          # 리쥼 펄스 활성 상태 (저속에서만 유지)
          resume_active = auto_longcontrol and (resume_pulse > 0 and CS.out.vEgo < 3.5) and lead_ok  # 내차 3.5m/s(≈12.6km/h)까지 펄스 유지

          # Kans: 오토크루즈 / 오토리쥼 메인 분기
          if auto_cruise_enabled and self._pending_activateCruise and not CS.out.cruiseState.enabled:
            # Kans: AutoCruise (0.25초 윈도 안에 최대 2회 버튼 시도)
            if not self.autoCruise_activate:
              self.autoCruise_activate = True
              self.autoCruise_frame = self.frame
              self.autoCruise_try_count = 0

            if self.autoCruise_activate:
              within_window = (self.frame - self.autoCruise_frame) * DT_CTRL <= self.cruiseDelay_time  # 예: 0.25초

              if within_window and self.autoCruise_try_count < 2:
                if (self.frame - self.last_button_frame) * DT_CTRL >= 0.12:
                  btn = CruiseButtons.RES_ACCEL if CS.out.activateCruise == 1 else CruiseButtons.DECEL_SET
                  self.send_btn(CS, can_sends, btn)
                  self.last_button_frame = self.frame
                  self.autoCruise_try_count += 1

              # 종료 조건: 시간 초과 / 2회 시도 완료 / 크루즈 실제 ON
              if (not within_window) or (self.autoCruise_try_count >= 2) or CS.out.cruiseState.enabled:
                self.autoCruise_activate = False
                self.autoCruise_frame = 0
                self.autoCruise_try_count = 0
                self._pending_activateCruise = False

          elif auto_engage_enabled and actuators.longControlState == LongCtrlState.starting:
            # Kans: SNG AutoResume: 1st step: 브레이크 펄스 (ActivateCruiseAfterBrake 플래그 세팅)
            ready_brake = (self.resume_fault_guard == 0) or CS.out.cruiseState.enabled
            if CC.longActive and not CS.out.brakePressed and not self.activateCruise_after_brake and not resume_active and ready_brake:
              self._brk_rc = (self._brk_rc + 1) & 0x3
              brk_idx = self._brk_rc
              apply_brake = self.brake_input(-self.brake_strength())
              # 브레이크신호 전송(롱컨 임시해제)
              can_sends.append(gmcan.create_brake_command(self.packer_ch, friction_brake_bus, apply_brake, brk_idx))
              Params().put_bool_nonblocking("ActivateCruiseAfterBrake", True)  # cruise.py에 브레이크 ON신호 전달
              self.activateCruise_after_brake = True  # 브레이크신호 초기화
              friction_sent_this_tick = True

            # 2nd step: AutoResume (RES 버튼 스팸)
            # 리쥼윈도 시작(기존윈도 없거나 이전윈도 닫힌=self.resume_activate=True 후 재진입용)
            if self.resume_frame == 0 or self.resume_activate:
              self.resume_frame = self.frame
              self.resume_activate = False
              self.resume_fault_guard = 0  # fault 방지용 카운터 초기화
              # 새 리쥼윈도 시작 시, 바로 RES 전송 가능하도록 버튼 간격 초기화
              self.last_button_frame = self.frame - int(0.12 / DT_CTRL)

            if not self.resume_activate:
              # Cruise fault 예방: 잦은 버튼송신 제한
              ready = (self.resume_fault_guard == 0) or CS.out.cruiseState.enabled
              if ready and (self.resume_fault_guard < 2):
                # 버튼 주기 0.12초, 리쥼실패율 가장 낮은 값으로 보임.
                if ((CS.lead_speed > 0.5) or (not CS.out.standstill)) and (self.frame - self.last_button_frame) * DT_CTRL >= 0.12:
                  self.send_btn(CS, can_sends, CruiseButtons.RES_ACCEL)
                  self.last_button_frame = self.frame
                  self.resume_fault_guard += 1  # 송신횟수 기록

              # 리쥼버튼 중단까지 지연시간(0.16~0.20)
              if (self.frame - self.resume_frame) * DT_CTRL >= self.resumeDelay_time:
                self.resume_activate = True

          else:
            # starting, auto_engage_enabled이 아니거나, 오토크루즈/오토리쥼 분기 모두 해당 안 될 때: 상태 정리
            self.activateCruise_after_brake = False

            if auto_engage_enabled:  # 오토리쥼이 진행중이면
              # 리쥼윈도 재개용 대기시간(resumeDelay_time * 1.5)
              if self.resume_frame > 0 and (self.frame - self.resume_frame) * DT_CTRL > (self.resumeDelay_time * 1.5):
                self.resume_frame = 0
                self.resume_activate = False
                self.resume_fault_guard = 0

            # 오토크루즈 초기화도 여기서(오토크루즈 분기 진입 안 했을 때)
            self.autoCruise_try_count = 0
            self.autoCruise_frame = 0
            self.autoCruise_activate = False

          # Kans: 실제 가스 송신 (언덕/평지 펄스 + 정상 가스)
          send_gas = self.apply_gas
          if resume_active:
            send_gas = resume_pulse
            at_full_stop = False
            acc_engaged = True
            can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, send_gas, idx, acc_engaged, at_full_stop, self.CP, resume_pulse=resume_pulse))
          else:
            # GasRegenCmdActive needs to be 1 to avoid cruise faults. It describes the ACC state, not actuation
            can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_gas, idx, acc_engaged, at_full_stop, self.CP, resume_pulse=resume_pulse))

          # Kans: 정규 브레이크 로직
          if not friction_sent_this_tick:
            self._brk_rc = (self._brk_rc + 1) & 0x3
            brk_idx_base = self._brk_rc
            can_sends.append(gmcan.create_friction_brake_command(self.packer_ch, friction_brake_bus, self.apply_brake, brk_idx_base, CC.enabled, near_stop, at_full_stop, self.CP))
            friction_sent_this_tick = True

          # Send dashboard UI commands (ACC status)
          send_fcw = hud_alert == VisualAlert.fcw
          can_sends.append(gmcan.create_acc_dashboard_command(self.packer_pt, CanBus.POWERTRAIN, CC.enabled,
                                                              hud_v_cruise * CV.MS_TO_KPH, hud_control, send_fcw))
      else:
        # to keep accel steady for logs when not sending gas
        accel += self.accel_g

      # Radar needs to know current speed and yaw rate (50hz),
      # and that ADAS is alive (10hz)
      if not self.CP.radarUnavailable and self.CP.networkLocation != NetworkLocation.fwdCamera and self.CP.carFingerprint not in SDGM_CAR:
        tt = self.frame * DT_CTRL
        time_and_headlights_step = 10
        if self.frame % time_and_headlights_step == 0:
          idx = (self.frame // time_and_headlights_step) % 4
          can_sends.append(gmcan.create_adas_time_status(CanBus.OBSTACLE, int((tt - self.start_time) * 60), idx))
          can_sends.append(gmcan.create_adas_headlights_status(self.packer_obj, CanBus.OBSTACLE))

        speed_and_accelerometer_step = 2
        if self.frame % speed_and_accelerometer_step == 0:
          idx = (self.frame // speed_and_accelerometer_step) % 4
          can_sends.append(gmcan.create_adas_steering_status(CanBus.OBSTACLE, idx))
          can_sends.append(gmcan.create_adas_accelerometer_speed_status(CanBus.OBSTACLE, abs(CS.out.vEgo), idx))

      if self.CP.networkLocation == NetworkLocation.gateway and self.frame % self.params.ADAS_KEEPALIVE_STEP == 0:
        can_sends += gmcan.create_adas_keepalive(CanBus.POWERTRAIN)

    else:
      # While car is braking, cancel button causes ECM to enter a soft disable state with a fault status.
      # A delayed cancellation allows camera to cancel and avoids a fault when user depresses brake quickly
      self.cancel_counter = self.cancel_counter + 1 if CC.cruiseControl.cancel else 0

      # 오토크루즈 '진입시도'중엔 CANCEL 송신금지
      auto_cruise_trying = (CS.out.activateCruise and not CS.out.cruiseState.enabled)

      # Stock longitudinal, integrated at camera
      if (self.frame - self.last_button_frame) * DT_CTRL >= 0.04:
        if self.cancel_counter > CAMERA_CANCEL_DELAY_FRAMES and (not auto_cruise_trying):
          self.last_button_frame = self.frame
          self.send_btn(CS, can_sends, CruiseButtons.CANCEL)

    if self.CP.networkLocation == NetworkLocation.fwdCamera:
      # Silence "Take Steering" alert sent by camera, forward PSCMStatus with HandsOffSWlDetectionStatus=1
      if self.frame % 20 == 0:
        can_sends.append(gmcan.create_pscm_status(self.packer_pt, CanBus.CAMERA, CS.pscm_status))

    new_actuators = actuators.as_builder()
    new_actuators.accel = accel
    new_actuators.torque = self.apply_torque_last / self.params.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    new_actuators.gas = self.apply_gas
    new_actuators.brake = self.apply_brake
    new_actuators.speed = self.apply_speed # kans: button spam

    self.frame += 1
    return new_actuators, can_sends

  # GM: AutoResume
  def brake_input(self, brake_force):
    MAX_BRAKE = 400

    if brake_force > 0.0:
      raise ValueError("brake_force는 0.0이하라야 됨.")

    scaled = int(-brake_force * 100)
    return max(0, min(MAX_BRAKE, scaled))

  def send_btn(self, CS, can_sends, cruise_btn, bus=None):
    if bus is None:
      if self.CP.carFingerprint in SDGM_CAR:
        bus = CanBus.POWERTRAIN
      elif self.CP.networkLocation == NetworkLocation.fwdCamera:
        bus = CanBus.CAMERA
      else:
        bus = CanBus.POWERTRAIN

    if bus == CanBus.CAMERA:
      if self.btn_rc_cam < 0:
        self.btn_rc_cam = int(CS.buttons_counter) & 0x3
      self.btn_rc_cam = (self.btn_rc_cam + 1) & 0x3
      rc = self.btn_rc_cam
    elif bus == CanBus.POWERTRAIN:
      if self.btn_rc_pt < 0:
        self.btn_rc_pt = int(CS.buttons_counter) & 0x3
      self.btn_rc_pt = (self.btn_rc_pt + 1) & 0x3
      rc = self.btn_rc_pt
    else:
      raise ValueError(f"Unsupported bus: {bus}")
    can_sends.append(gmcan.create_buttons(self.packer_pt, bus, rc, cruise_btn))

  def brake_strength(self) -> float:
    if self.CP.carFingerprint in EV_CAR or self.CP.carFingerprint in SDGM_CAR:
      return 0.4
    else:
      return 0.7
