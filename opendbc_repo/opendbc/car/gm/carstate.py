import copy
from opendbc.can import CANDefine, CANParser
from cereal import car
from openpilot.common.params import Params #kans
import numpy as np
from opendbc.car import Bus, create_button_events, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.gm.values import DBC, AccState, CruiseButtons, STEER_THRESHOLD, CAR, GMFlags, \
   CC_ONLY_CAR, CAMERA_ACC_CAR, EV_CAR, SDGM_CAR
import cereal.messaging as messaging

ButtonType = structs.CarState.ButtonEvent.Type
TransmissionType = structs.CarParams.TransmissionType
NetworkLocation = structs.CarParams.NetworkLocation
GearShifter = structs.CarState.GearShifter
STANDSTILL_THRESHOLD = 10 * 0.0311 * CV.KPH_TO_MS

BUTTONS_DICT = {CruiseButtons.RES_ACCEL: ButtonType.accelCruise, CruiseButtons.DECEL_SET: ButtonType.decelCruise,
                CruiseButtons.MAIN: ButtonType.mainCruise, CruiseButtons.CANCEL: ButtonType.cancel,
                CruiseButtons.GAP_DIST: ButtonType.gapAdjustCruise}

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])
    self.shifter_values = can_define.dv["ECMPRDNL2"]["PRNDL2"]
    self.cluster_speed_hyst_gap = CV.KPH_TO_MS / 2.
    self.cluster_min_speed = CV.KPH_TO_MS / 2.

    self.loopback_lka_steering_cmd_updated = False
    self.loopback_lka_steering_cmd_ts_nanos = 0
    self.pt_lka_steering_cmd_counter = 0
    self.cam_lka_steering_cmd_counter = 0
    self.buttons_counter = 0
    self.single_pedal_mode = False
    self.pedal_steady = 0.
    self.cruise_buttons = 0
    # GAP_DIST
    self.distance_button = 0
    # Kans: ambient temperature (°C)
    self.ambient_c = 0.0
    # Kans: lead_car condition
    self.lead_distance = float('inf')
    self.lead_speed = 0.0
    self.lead_accel = 0.0
    self.sm = messaging.SubMaster(['radarState', 'deviceState'])

    self.cruiseMain_on = False
    # accFault hyst
    self._standstill_hyst = True
    self._ss_enter = 0.12  # (.086=STANDSTILL_THRESHOLD = 10기준) .086->.12로 상향. 더 높은 속도아래에서 정지 판단
    self._ss_exit = 0.25  # m/s, 스탠드스틸에서 '나갈' 때 임계. .15->.25로 상향. 더 높은 속도위에서 출발 판단.
    self._creep_max = 0.45  # m/s, 크리핑 윈도 (≈1.08 km/h)
    # Kans: TPMS
    self.KPA_TO_PSI = 0.1450377377
    self.TPMS_GAIN = 1.125
    self.TPMS_OFFSET = 0.5
    # for delay Accfault event
    self.accFaultedCount = 0

  def kpa_to_psi(self, kpa_g: float) -> float:
    return float(kpa_g) * self.KPA_TO_PSI

  def psi_display(self, psi_raw: float) -> float:
    return self.TPMS_GAIN * psi_raw - self.TPMS_OFFSET

  def update_button_enable(self, buttonEvents: list[structs.CarState.ButtonEvent]):
    if not self.CP.pcmCruise:
      for b in buttonEvents:
        # The ECM allows enabling on falling edge of set, but only rising edge of resume
        if (b.type == ButtonType.accelCruise and b.pressed) or \
          (b.type == ButtonType.decelCruise and not b.pressed):
          return True
    return False

  def update(self, can_parsers) -> structs.CarState:
    # lead_car condition(dRel, vLead extract)
    self.sm.update(0)
    if self.sm.updated['radarState']:
      lead = self.sm['radarState'].leadOne
      if lead is not None and lead.status:
        self.lead_distance = float(lead.dRel if lead.dRel is not None else float('inf'))
        self.lead_speed = float(lead.vLead if lead.vLead is not None else 0.0)
        self.lead_accel = float(lead.aLead if lead.aLead is not None else 0.0)
      else:
        self.lead_distance = float('inf')
        self.lead_speed = 0.0
        self.lead_accel = 0.0
    # ambient temperature 업데이트
    if self.sm.updated['deviceState']:
      ds = self.sm['deviceState']
      # ambientTempCDEPRECATED가 None일 수도 있으니
      if ds.ambientTempCDEPRECATED is not None:
        self.ambient_c = float(ds.ambientTempCDEPRECATED)

    pt_cp = can_parsers[Bus.pt]
    cam_cp = can_parsers[Bus.cam]
    loopback_cp = can_parsers[Bus.loopback]

    ret = structs.CarState()

    prev_cruise_buttons = self.cruise_buttons
    prev_distance_button = self.distance_button
    self.cruise_buttons = pt_cp.vl["ASCMSteeringButton"]["ACCButtons"]
    self.distance_button = pt_cp.vl["ASCMSteeringButton"]["DistanceButton"]
    self.buttons_counter = pt_cp.vl["ASCMSteeringButton"]["RollingCounter"]

    self.pscm_status = copy.copy(pt_cp.vl["PSCMStatus"])
    # GAP_DIST
    if self.cruise_buttons in [CruiseButtons.UNPRESS, CruiseButtons.INIT] and self.distance_button:
      self.cruise_buttons = CruiseButtons.GAP_DIST

    if self.CP.enableBsm:
      ret.leftBlindspot = pt_cp.vl["BCMBlindSpotMonitor"]["LeftBSM"] == 1
      ret.rightBlindspot = pt_cp.vl["BCMBlindSpotMonitor"]["RightBSM"] == 1

    # Variables used for avoiding LKAS faults
    self.loopback_lka_steering_cmd_updated = len(loopback_cp.vl_all["ASCMLKASteeringCmd"]["RollingCounter"]) > 0
    if self.loopback_lka_steering_cmd_updated:
      self.loopback_lka_steering_cmd_ts_nanos = loopback_cp.ts_nanos["ASCMLKASteeringCmd"]["RollingCounter"]
    if self.CP.networkLocation == NetworkLocation.fwdCamera and not self.CP.flags & GMFlags.NO_CAMERA.value:
      self.pt_lka_steering_cmd_counter = pt_cp.vl["ASCMLKASteeringCmd"]["RollingCounter"]
      self.cam_lka_steering_cmd_counter = cam_cp.vl["ASCMLKASteeringCmd"]["RollingCounter"]

    # This is to avoid a fault where you engage while still moving backwards after shifting to D.
    # An Equinox has been seen with an unsupported status (3), so only check if either wheel is in reverse (2)
    left_whl_sign = -1 if pt_cp.vl["EBCMWheelSpdRear"]["RLWheelDir"] == 2 else 1
    right_whl_sign = -1 if pt_cp.vl["EBCMWheelSpdRear"]["RRWheelDir"] == 2 else 1
    ret.wheelSpeeds = self.get_wheel_speeds(
      left_whl_sign * pt_cp.vl["EBCMWheelSpdFront"]["FLWheelSpd"],
      right_whl_sign * pt_cp.vl["EBCMWheelSpdFront"]["FRWheelSpd"],
      left_whl_sign * pt_cp.vl["EBCMWheelSpdRear"]["RLWheelSpd"],
      right_whl_sign * pt_cp.vl["EBCMWheelSpdRear"]["RRWheelSpd"],
    )
    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    # sample rear wheel speeds, standstill=True if ECM allows engagement with brake
    # accFault hyst
    v_rl = abs(ret.wheelSpeeds.rl)
    v_rr = abs(ret.wheelSpeeds.rr)
    if self._standstill_hyst:
      if (v_rl > self._ss_exit) or (v_rr > self._ss_exit):
        self._standstill_hyst = False
    else:
      if (v_rl < self._ss_enter) and (v_rr < self._ss_enter):
        self._standstill_hyst = True
    ret.standstill = self._standstill_hyst  # 기존 ret.standstill 대체
    #ret.standstill = abs(ret.wheelSpeeds.rl) <= STANDSTILL_THRESHOLD and abs(ret.wheelSpeeds.rr) <= STANDSTILL_THRESHOLD

    if pt_cp.vl["ECMPRDNL2"]["ManualMode"] == 1:
      ret.gearShifter = self.parse_gear_shifter("T")
    else:
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(pt_cp.vl["ECMPRDNL2"]["PRNDL2"], None))

    if self.CP.flags & GMFlags.NO_ACCELERATOR_POS_MSG.value:
      ret.brake = pt_cp.vl.get("EBCMBrakePedalPosition", {}).get("BrakePedalPosition", 0) / 0xd0
    else:
      ret.brake = pt_cp.vl.get("ECMAcceleratorPos", {}).get("BrakePedalPos", 0)
    if (self.CP.flags & GMFlags.FORCE_BRAKE_C9.value) or \
       ((self.CP.networkLocation == NetworkLocation.fwdCamera) and
        (self.CP.carFingerprint not in SDGM_CAR)):
      ret.brakePressed = pt_cp.vl["ECMEngineStatus"]["BrakePressed"] != 0
    else:
      # Some Volt 2016-17 have loose brake pedal push rod retainers which causes the ECM to believe
      # that the brake is being intermittently pressed without user interaction.
      # To avoid a cruise fault we need to use a conservative brake position threshold
      # https://static.nhtsa.gov/odi/tsbs/2017/MC-10137629-9999.pdf
      analog_thresh = 0.07 if (self.CP.flags & GMFlags.NO_ACCELERATOR_POS_MSG.value) else 8
      ret.brakePressed = ret.brake >= analog_thresh

    # Regen braking is braking
    if self.CP.transmissionType == TransmissionType.direct:
      ret.regenBraking = pt_cp.vl["EBCMRegenPaddle"]["RegenPaddle"] != 0
      self.single_pedal_mode = ret.gearShifter == GearShifter.low or pt_cp.vl["EVDriveMode"]["SinglePedalModeActive"] == 1 or (ret.regenBraking and GearShifter.manumatic) or (self.CP.carFingerprint in [CAR.CHEVROLET_BOLT_EUV, CAR.CHEVROLET_BOLT_CC] and self.CP.enableGasInterceptorDEPRECATED)

    # kans: TPMS
    if self.CP.flags & GMFlags.TPMS_MSG.value:
      kpa_fl = float(pt_cp.vl["TPMS"]["PRESSURE_FL"])
      kpa_fr = float(pt_cp.vl["TPMS"]["PRESSURE_FR"])
      kpa_rl = float(pt_cp.vl["TPMS"]["PRESSURE_RL"])
      kpa_rr = float(pt_cp.vl["TPMS"]["PRESSURE_RR"])

      # 온도(°C)에 따른 보정계수
      ambient_c = self.ambient_c
      if ambient_c <= 3.0:
        temp_gain = 1.8
      if ambient_c <= 6.0:
        temp_gain = 1.6
      elif ambient_c <= 10.0:
        temp_gain = 1.1
      elif ambient_c <= 14.0:
        temp_gain = 1.07
      elif ambient_c <= 20.0:
        temp_gain = 1.0
      elif ambient_c <= 25.0:
        temp_gain = 0.98
      elif ambient_c <= 30.0:
        temp_gain = 0.96
      elif ambient_c <= 32.0:
        temp_gain = 0.94
      elif ambient_c <= 34.0:
        temp_gain = 0.92
      if ambient_c <= 36.0:
        temp_gain = 0.9
      elif ambient_c <= 40.0:
        temp_gain = 0.88
      else:
        temp_gain = 0.86  # 극고온 fallback

      psi_fl = self.psi_display(self.kpa_to_psi(kpa_fl) * temp_gain)
      psi_fr = self.psi_display(self.kpa_to_psi(kpa_fr) * temp_gain)
      psi_rl = self.psi_display(self.kpa_to_psi(kpa_rl) * temp_gain)
      psi_rr = self.psi_display(self.kpa_to_psi(kpa_rr) * temp_gain)

      ret.tpms.fl = psi_fl
      ret.tpms.fr = psi_fr
      ret.tpms.rl = psi_rl
      ret.tpms.rr = psi_rr

    if self.CP.enableGasInterceptorDEPRECATED:
      ret.gas = (pt_cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS"] + pt_cp.vl["GAS_SENSOR"]["INTERCEPTOR_GAS2"]) / 2.
      # Panda 515 threshold = 10.88. Set lower to avoid panda blocking messages and GasInterceptor faulting.
      threshold = 23 if self.CP.carFingerprint in CAMERA_ACC_CAR else 4
      ret.gasPressed = ret.gas > threshold
    else:
      ret.gas = pt_cp.vl["AcceleratorPedal2"]["AcceleratorPedal2"] / 254.
      ret.gasPressed = ret.gas > 0  # 1e-5

    ret.steeringAngleDeg = pt_cp.vl["PSCMSteeringAngle"]["SteeringWheelAngle"]
    ret.steeringRateDeg = pt_cp.vl["PSCMSteeringAngle"]["SteeringWheelRate"]
    ret.steeringTorque = pt_cp.vl["PSCMStatus"]["LKADriverAppldTrq"]
    ret.steeringTorqueEps = pt_cp.vl["PSCMStatus"]["LKATorqueDelivered"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD

    # 0 inactive, 1 active, 2 temporarily limited, 3 failed
    self.lkas_status = pt_cp.vl["PSCMStatus"]["LKATorqueDeliveredStatus"]
    ret.steerFaultTemporary = self.lkas_status == 2
    ret.steerFaultPermanent = self.lkas_status == 3

    # 1 - open, 0 - closed
    ret.doorOpen = (pt_cp.vl["BCMDoorBeltStatus"]["FrontLeftDoor"] == 1 or
                    pt_cp.vl["BCMDoorBeltStatus"]["FrontRightDoor"] == 1 or
                    pt_cp.vl["BCMDoorBeltStatus"]["RearLeftDoor"] == 1 or
                    pt_cp.vl["BCMDoorBeltStatus"]["RearRightDoor"] == 1)

    # 1 - latched
    ret.seatbeltUnlatched = pt_cp.vl["BCMDoorBeltStatus"]["LeftSeatBelt"] == 0
    ret.leftBlinker = pt_cp.vl["BCMTurnSignals"]["TurnSignals"] == 1
    ret.rightBlinker = pt_cp.vl["BCMTurnSignals"]["TurnSignals"] == 2

    ret.parkingBrake = pt_cp.vl["BCMGeneralPlatformStatus"]["ParkBrakeSwActive"] == 1

    ecu_cruise_main = pt_cp.vl["ECMEngineStatus"]["CruiseMainOn"] != 0
    ret.cruiseState.available = ecu_cruise_main
    self.cruiseMain_on = ret.cruiseState.available

    ret.espDisabled = pt_cp.vl["ESPStatus"]["TractionControlOn"] != 1

    # accFault hyst
    creeping = max(v_rl, v_rr) < self._creep_max
    # Delay Accfault event & regarding not acc faulted while user's brake pedal
    cruise_fault = (pt_cp.vl["AcceleratorPedal2"]["CruiseState"] == AccState.FAULTED and not creeping)
    friction_unavailable = pt_cp.vl["EBCMFrictionBrakeStatus"]["FrictionBrakeUnavailable"]
    accFaulted = cruise_fault or (friction_unavailable == 1)
    if self.CP.carFingerprint == CAR.CHEVROLET_VOLT and (ret.brakePressed or ret.standstill or creeping):
      self.accFaultedCount = 0
      ret.accFaulted = False
    else:
      self.accFaultedCount = self.accFaultedCount + 1 if accFaulted else 0
      ret.accFaulted = True if self.accFaultedCount > 50 else False

    ret.cruiseState.enabled = pt_cp.vl["AcceleratorPedal2"]["CruiseState"] != AccState.OFF
    ret.cruiseState.standstill = pt_cp.vl["AcceleratorPedal2"]["CruiseState"] == AccState.STANDSTILL
    # kans: avoid to accFault
    if self.CP.carFingerprint not in CAR.CHEVROLET_VOLT:
      ret.cruiseState.standstill = False
    if self.CP.networkLocation == NetworkLocation.fwdCamera and not self.CP.flags & GMFlags.NO_CAMERA.value:
      if self.CP.carFingerprint not in CC_ONLY_CAR:
        ret.cruiseState.speed = cam_cp.vl["ASCMActiveCruiseControlStatus"]["ACCSpeedSetpoint"] * CV.KPH_TO_MS
      ret.stockAeb = False
      # openpilot controls nonAdaptive when not pcmCruise
      if self.CP.pcmCruise and self.CP.carFingerprint not in CC_ONLY_CAR: 
        ret.cruiseState.nonAdaptive = cam_cp.vl["ASCMActiveCruiseControlStatus"]["ACCCruiseState"] not in (2, 3)
    if self.CP.carFingerprint in CC_ONLY_CAR:
      ret.accFaulted = False
      ret.cruiseState.speed = pt_cp.vl["ECMCruiseControl"]["CruiseSetSpeed"] * CV.KPH_TO_MS
      ret.cruiseState.enabled = pt_cp.vl["ECMCruiseControl"]["CruiseActive"] != 0
    prev_lkas_enabled = self.lkas_enabled
    self.lkas_enabled = pt_cp.vl["ASCMSteeringButton"]["LKAButton"]

    acc_status = pt_cp.vl["AcceleratorPedal2"]["CruiseState"]
    self.pcm_acc_status = acc_status
    ret.accStatus = int(acc_status)

    ret.vCluRatio = 1.0 if self.CP.carFingerprint in EV_CAR else 0.96

    # Don't add event if transitioning from INIT, unless it's to an actual button
    if self.cruise_buttons != CruiseButtons.UNPRESS or prev_cruise_buttons != CruiseButtons.INIT:
      ret.buttonEvents = [
        *create_button_events(self.cruise_buttons, prev_cruise_buttons, BUTTONS_DICT,
                              unpressed_btn=CruiseButtons.UNPRESS),
        *create_button_events(self.distance_button, prev_distance_button,
                              {1: ButtonType.gapAdjustCruise}),
        *create_button_events(self.lkas_enabled, prev_lkas_enabled,
                              {1: ButtonType.lkas})
      ]

    return ret

  @staticmethod
  def get_can_parsers(CP):
    pt_messages = []

    if CP.flags & GMFlags.TPMS_MSG.value:
      pt_messages.append(("TPMS", 5))

    if CP.networkLocation == NetworkLocation.fwdCamera:
      pt_messages += [
        ("ASCMLKASteeringCmd", float('nan')),
      ]
    if CP.transmissionType == TransmissionType.direct:
      pt_messages += [
        ("EBCMRegenPaddle", 50),
        ("EVDriveMode", float('nan')),
      ]
    loopback_messages = [
      ("ASCMLKASteeringCmd", float('nan')),
    ]

    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, 0),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 2),
      Bus.loopback: CANParser(DBC[CP.carFingerprint][Bus.pt], loopback_messages, 128),
    }

