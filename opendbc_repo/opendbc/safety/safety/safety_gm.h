#pragma once

#include "safety_declarations.h"

// TODO: do checksum and counter checks. Add correct timestep, 0.1s for now.
#define GM_COMMON_RX_CHECKS \
    {.msg = {{0x184, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}}, \
    {.msg = {{0x34A, 0, 5, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}}, \
    {.msg = {{0x1E1, 0, 7, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}}, \
    {.msg = {{0x1C4, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}}, \
    {.msg = {{0xC9, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}}, \

#define GM_ACC_RX_CHECKS \
    {.msg = {{0xBE, 0, 6, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U},    /* Volt, Silverado, Acadia Denali */ \
             {0xBE, 0, 7, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U},    /* Bolt EUV */ \
             {0xBE, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}}},  /* Escalade */ \

static const LongitudinalLimits *gm_long_limits;

enum {
  GM_BTN_UNPRESS = 1,
  GM_BTN_RESUME = 2,
  GM_BTN_SET = 3,
  GM_BTN_MAIN = 5,
  GM_BTN_CANCEL = 6,
};

typedef enum {
  GM_ASCM,
  GM_CAM
} GmHardware;
static GmHardware gm_hw = GM_ASCM;
static bool gm_cam_long = false;
static bool gm_pcm_cruise = false;
static bool gm_has_acc = true;
static bool gm_pedal_long = false;
static bool gm_cc_long = false;
static bool gm_force_ascm = false;
static int skip_brake_disable_frame = 0; //리쥼브레크신호 지연 프레임

const int GM_STANDSTILL_THRSLD = 10;  // 0.311kph 미세 속도에도 standstill해제될 가능성 있으므로 20~15로 튜닝해볼 필요 있음.
const int GM_GAS_INTERCEPTOR_THRESHOLD = 550;
#define GM_GET_INTERCEPTOR(msg) (((GET_BYTE((msg), 0) << 8) + GET_BYTE((msg), 1) + (GET_BYTE((msg), 2) << 8) + GET_BYTE((msg), 3)) / 2U)

static void handle_gm_wheel_buttons(const CANPacket_t *to_push) {
  static int frame = 0;  //리쥼브레크신호 지연 프레임
  frame++;

  int button = (GET_BYTE(to_push, 5) & 0x70U) >> 4;

  // enter controls on falling edge of set or rising edge of resume (avoids fault)
  bool set = (cruise_button_prev == GM_BTN_SET) && (button != GM_BTN_SET);
  bool res = (button == GM_BTN_RESUME) && (cruise_button_prev != GM_BTN_RESUME);
  if (set || res) {
    controls_allowed = true;
  }

  // exit controls on cancel press
  if (button == GM_BTN_CANCEL) {
    controls_allowed = false;
  }
  // 오토리쥼 브레이크 지연프레임(0.1초)
  if (res) {
    skip_brake_disable_frame = frame + 10;
  }
  cruise_button_prev = button;
}

static void gm_rx_hook(const CANPacket_t *to_push) {
  static int frame = 0;  //브레크신호 프레임 초기화
  frame++;

  if (GET_BUS(to_push) == 0U) {
    int addr = GET_ADDR(to_push);

    if (addr == 0x184) {
      int torque_driver_new = ((GET_BYTE(to_push, 6) & 0x7U) << 8) | GET_BYTE(to_push, 7);
      torque_driver_new = to_signed(torque_driver_new, 11);
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // sample rear wheel speeds
    if (addr == 0x34A) {
      int left_rear_speed = (GET_BYTE(to_push, 0) << 8) | GET_BYTE(to_push, 1);
      int right_rear_speed = (GET_BYTE(to_push, 2) << 8) | GET_BYTE(to_push, 3);
      vehicle_moving = (left_rear_speed > GM_STANDSTILL_THRSLD) || (right_rear_speed > GM_STANDSTILL_THRSLD);
    }

    // ACC steering wheel buttons (GM_CAM is tied to the PCM)
    if ((addr == 0x1E1) && (!gm_pcm_cruise || gm_cc_long)) {
      handle_gm_wheel_buttons(to_push);
    }

    // Reference for brake pressed signals:
    // https://github.com/commaai/openpilot/blob/master/selfdrive/car/gm/carstate.py
    if ((addr == 0xBE) && (gm_hw == GM_ASCM)) {
      brake_pressed = GET_BYTE(to_push, 1) >= 10U;
    }

    if (addr == 0xC9) {
      if (gm_hw == GM_CAM) {
        brake_pressed = (GET_BYTE(to_push, 5) & 0x01U) != 0U;  // CAM_ACC용 브레이크on/off 체크(201핑거 40번째 비트)
      }
      acc_main_on = (GET_BYTE(to_push, 3) & 0x20U) != 0U;  // 크루즈 메인스위치 체크(201핑거 29번째 비트)
    }

    // 프레임지연 동안(+정지중) 크루즈폴트 무시
    if (frame > skip_brake_disable_frame) {
      if (brake_pressed && !brake_pressed_prev && vehicle_moving) {
        controls_allowed = false;
      }
    }
    brake_pressed_prev = brake_pressed;

    if (addr == 0x1C4) {
      if (!enable_gas_interceptor) {
        gas_pressed = GET_BYTE(to_push, 5) != 0U;
      }

      // enter controls on rising edge of ACC, exit controls when ACC off
      if (gm_pcm_cruise && gm_has_acc) {
        bool cruise_engaged = (GET_BYTE(to_push, 1) >> 5) != 0U;
        // 이전 상태 저장
        bool prev = cruise_engaged_prev;
        // 기존 stock ACC 토글 로직
        pcm_cruise_check(cruise_engaged);
        // Rising edge(Off→Active) 시점에 허용
        if (cruise_engaged && !prev) {
          controls_allowed = true;
        }
        // 상태 갱신
        cruise_engaged_prev = cruise_engaged;
      }
    }

    // Cruise check for CC only cars
    if ((addr == 0x3D1) && !gm_has_acc) {
      bool cruise_engaged = (GET_BYTE(to_push, 4) >> 7) != 0U;
      if (gm_cc_long) {
        pcm_cruise_check(cruise_engaged);
      } else {
        cruise_engaged_prev = cruise_engaged;
      }
    }

    if (addr == 0xBD) {
      regen_braking = (GET_BYTE(to_push, 0) >> 4) != 0U;
    }

    // Pedal Interceptor
    if ((addr == 0x201) && enable_gas_interceptor) {
      int gas_interceptor = GM_GET_INTERCEPTOR(to_push);
      gas_pressed = gas_interceptor > GM_GAS_INTERCEPTOR_THRESHOLD;
      gas_interceptor_prev = gas_interceptor;
      // gm_pcm_cruise = false;
    }

    bool stock_ecu_detected = (addr == 0x180);  // ASCMLKASteeringCmd

    // Check ASCMGasRegenCmd only if we're blocking it
    if (!gm_pcm_cruise && !gm_pedal_long && (addr == 0x2CB)) {
      stock_ecu_detected = true;
    }
    // 운전자 가스오버라이드에도 롱컨 유지
    alternative_experience |= ALT_EXP_DISABLE_DISENGAGE_ON_GAS;
    generic_rx_checks(stock_ecu_detected);
  }
}

static bool gm_tx_hook(const CANPacket_t *to_send) {
  const TorqueSteeringLimits GM_STEERING_LIMITS = {
    .max_steer = 300,
    .max_rate_up = 10,
    .max_rate_down = 15,
    .driver_torque_allowance = 65,
    .driver_torque_multiplier = 4,
    .max_rt_delta = 128,
    .max_rt_interval = 250000,
    .type = TorqueDriverLimited,
  };

  bool tx = true;
  int addr = GET_ADDR(to_send);

  // BRAKE: safety check
  if (addr == 0x315) {
    int brake = ((GET_BYTE(to_send, 0) & 0xFU) << 8) + GET_BYTE(to_send, 1);
    brake = (0x1000 - brake) & 0xFFF;
    if (longitudinal_brake_checks(brake, *gm_long_limits)) {
      tx = false;
    }
  }

  // LKA STEER: safety check
  if (addr == 0x180) {
    int desired_torque = ((GET_BYTE(to_send, 0) & 0x7U) << 8) + GET_BYTE(to_send, 1);
    desired_torque = to_signed(desired_torque, 11);

    bool steer_req = GET_BIT(to_send, 3U);

    if (steer_torque_cmd_checks(desired_torque, steer_req, GM_STEERING_LIMITS)) {
      //tx = false;
    }
  }

  // GAS: safety check (interceptor)
  if (addr == 0x200) {
    if (longitudinal_interceptor_checks(to_send)) {
      tx = 0;
    }
  }

  // GAS/REGEN: safety check
  if (addr == 0x2CB) {
    bool apply = GET_BIT(to_send, 0U);
    if (apply) {
      if(!controls_allowed) print("@@auto cruise control enabled....\n");
        controls_allowed = true;        
    }
    int gas_regen = (((GET_BYTE(to_send, 1) & 0x7U) << 16) | (GET_BYTE(to_send, 2) << 8) | GET_BYTE(to_send, 3)) - 180272U;

    bool violation = false;
    // Allow apply bit in pre-enabled and overriding states
    //violation |= !controls_allowed && apply;
    violation |= longitudinal_gas_checks(gas_regen, *gm_long_limits);

    if (violation) {
      tx = false;
    }
  }

  // BUTTONS: used for resume spamming and cruise cancellation with stock longitudinal
  if ((addr == 0x1E1) && (gm_pcm_cruise || gm_pedal_long || gm_cc_long)) {
    int button = (GET_BYTE(to_send, 5) >> 4) & 0x7U;

    bool allowed_btn = (button == GM_BTN_CANCEL) && cruise_engaged_prev;
    // For standard CC, allow spamming of SET / RESUME
    if (gm_cc_long) {
      allowed_btn |= cruise_engaged_prev && ((button == GM_BTN_SET) || (button == GM_BTN_RESUME) || (button == GM_BTN_UNPRESS));
    }

    if (!allowed_btn) {
      tx = false;
    }
  }
  return tx;
}

static int gm_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  // ASCM은 어떤 메시지도 포워딩하지 않음
  if ((gm_hw == GM_ASCM) || gm_cc_long || gm_cam_long) {
    bus_fwd = -1;
  }

  if ((gm_hw == GM_CAM) || gm_cam_long) {
    if (bus_num == 0) {
      // block PSCMStatus; forwarded through openpilot to hide an alert from the camera
      bool is_pscm_msg = (addr == 0x184);
      if (!is_pscm_msg) {
        bus_fwd = 2;
      }
    }

    if (bus_num == 2) {
      // block lkas message and acc messages if gm_cam_long, forward all others
      bool is_lkas_msg = (addr == 0x180);
      bool is_acc_msg = (addr == 0x315) || (addr == 0x2CB) || (addr == 0x370);
      bool block_msg = is_lkas_msg || (is_acc_msg && gm_cam_long);
      if (!block_msg) {
        bus_fwd = 0;
      }
    }
  }

  return bus_fwd;
}

static safety_config gm_init(uint16_t param) {
  const uint16_t GM_PARAM_HW_CAM = 1;
  const uint16_t GM_PARAM_HW_CAM_LONG = 2;
  const uint16_t GM_PARAM_EV = 4;
  const uint16_t GM_PARAM_CC_LONG = 8;
  const uint16_t GM_PARAM_HW_ASCM_LONG = 16;
  const uint16_t GM_PARAM_NO_ACC = 32;
  const uint16_t GM_PARAM_PEDAL_LONG = 64;  // TODO: this can be inferred
  const uint16_t GM_PARAM_PEDAL_INTERCEPTOR = 128;

  // common safety checks assume unscaled integer values
  static const int GM_GAS_TO_CAN = 8;  // 1 / 0.125

  static const LongitudinalLimits GM_ASCM_LONG_LIMITS = {
    .max_gas = 1018 * GM_GAS_TO_CAN,
    .min_gas = -650 * GM_GAS_TO_CAN,
    .inactive_gas = -650 * GM_GAS_TO_CAN,
    .max_brake = 400,
  };

  static const CanMsg GM_ASCM_TX_MSGS[] = {{0x180, 0, 4}, {0x409, 0, 7}, {0x40A, 0, 7}, {0x2CB, 0, 8}, {0x370, 0, 6}, {0x200, 0, 6}, {0x1E1, 0, 7}, {0xBD, 0, 7},  // pt bus
                                           {0xA1, 1, 7}, {0x306, 1, 8}, {0x308, 1, 7}, {0x310, 1, 2},   // obs bus
                                           {0x315, 2, 5}, {0x1E1, 2, 7}};  // ch bus


  static const LongitudinalLimits GM_CAM_LONG_LIMITS = {
    .max_gas = 1346 * GM_GAS_TO_CAN,
    .min_gas = -540 * GM_GAS_TO_CAN,
    .inactive_gas = -500 * GM_GAS_TO_CAN,
    .max_brake = 400,
  };

  static const CanMsg GM_CAM_LONG_TX_MSGS[] = {{0x180, 0, 4}, {0x315, 0, 5}, {0x2CB, 0, 8}, {0x370, 0, 6}, {0x200, 0, 6}, {0x1E1, 0, 7},  // pt bus
                                               {0x315, 2, 5}, {0x2CB, 2, 8}, {0x184, 2, 8}, {0x1E1, 2, 7}};  // camera bus

  // TODO: do checksum and counter checks. Add correct timestep, 0.1s for now.
  static RxCheck gm_rx_checks[] = {
    GM_COMMON_RX_CHECKS
    GM_ACC_RX_CHECKS
  };

  static RxCheck gm_ev_rx_checks[] = {
    GM_COMMON_RX_CHECKS
    GM_ACC_RX_CHECKS
    {.msg = {{0xBD, 0, 7, .ignore_checksum = true, .ignore_counter = true, .frequency = 40U}, { 0 }, { 0 }}},
  };

  static RxCheck gm_no_acc_rx_checks[] = {
    GM_COMMON_RX_CHECKS
    {.msg = {{0x3D1, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}},  // Non-ACC PCM
  };

  static RxCheck gm_no_acc_ev_rx_checks[] = {
    GM_COMMON_RX_CHECKS
    {.msg = {{0xBD, 0, 7, .ignore_checksum = true, .ignore_counter = true, .frequency = 40U}, { 0 }, { 0 }}},
    {.msg = {{0x3D1, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}},  // Non-ACC PCM
  };

  static RxCheck gm_pedal_rx_checks[] = {
    GM_COMMON_RX_CHECKS
    {.msg = {{0xBD, 0, 7, .ignore_checksum = true, .ignore_counter = true, .frequency = 40U}, { 0 }, { 0 }}},
    {.msg = {{0x3D1, 0, 8, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}},  // Non-ACC PCM
    {.msg = {{0x201, 0, 6, .ignore_checksum = true, .ignore_counter = true, .frequency = 10U}, { 0 }, { 0 }}},  // pedal
  };

  static const CanMsg GM_CAM_TX_MSGS[] = {{0x180, 0, 4}, {0x1E1, 0, 7}, {0x200, 0, 6},  // pt bus
                                          {0x1E1, 2, 7}, {0x184, 2, 8}};  // camera bus


  static const CanMsg GM_CC_LONG_TX_MSGS[] = {{0x180, 0, 4}, {0x1E1, 0, 7},  // pt bus
                                              {0x184, 2, 8}, {0x1E1, 2, 7}};  // camera bus
  gm_hw = GET_FLAG(param, GM_PARAM_HW_CAM) ? GM_CAM : GM_ASCM;

  gm_force_ascm = GET_FLAG(param, GM_PARAM_HW_ASCM_LONG);

  if ((gm_hw == GM_ASCM) || gm_force_ascm) {
    gm_long_limits = &GM_ASCM_LONG_LIMITS;
  } else if (gm_hw == GM_CAM) {
    gm_long_limits = &GM_CAM_LONG_LIMITS;
  } else {
  }

  gm_pedal_long = GET_FLAG(param, GM_PARAM_PEDAL_LONG);
  gm_cc_long = GET_FLAG(param, GM_PARAM_CC_LONG);
  gm_cam_long = GET_FLAG(param, GM_PARAM_HW_CAM_LONG) && !gm_cc_long;
  gm_pcm_cruise = ((gm_hw == GM_CAM) && (!gm_cam_long || gm_cc_long) && !gm_force_ascm && !gm_pedal_long);

  gm_has_acc = !GET_FLAG(param, GM_PARAM_NO_ACC);
  enable_gas_interceptor = GET_FLAG(param, GM_PARAM_PEDAL_INTERCEPTOR);

  safety_config ret;
  if (gm_hw == GM_CAM) {
    // FIXME: cppcheck thinks that gm_cam_long is always false. This is not true
    // if ALLOW_DEBUG is defined but cppcheck is run without ALLOW_DEBUG
    // cppcheck-suppress knownConditionTrueFalse
    if (gm_cc_long) {
      ret = BUILD_SAFETY_CFG(gm_rx_checks, GM_CC_LONG_TX_MSGS);
    } else if (gm_cam_long) {
      ret = BUILD_SAFETY_CFG(gm_rx_checks, GM_CAM_LONG_TX_MSGS);
    } else {
      ret = BUILD_SAFETY_CFG(gm_rx_checks, GM_CAM_TX_MSGS);
    }
  } else {
    ret = BUILD_SAFETY_CFG(gm_rx_checks, GM_ASCM_TX_MSGS);
  }

  const bool gm_ev = GET_FLAG(param, GM_PARAM_EV);
  if (enable_gas_interceptor) {
    SET_RX_CHECKS(gm_pedal_rx_checks, ret);
  } else if (!gm_has_acc && gm_ev) {
    SET_RX_CHECKS(gm_no_acc_ev_rx_checks, ret);
  } else if (!gm_has_acc && !gm_ev) {
    SET_RX_CHECKS(gm_no_acc_rx_checks, ret);
  } else if (gm_ev) {
    SET_RX_CHECKS(gm_ev_rx_checks, ret);
  } else {}

  // ASCM does not any work
  if ((gm_hw == GM_ASCM) || gm_cc_long || gm_cam_long) {
  }
  return ret;
}

const safety_hooks gm_hooks = {
  .init = gm_init,
  .rx = gm_rx_hook,
  .tx = gm_tx_hook,
  .fwd = gm_fwd_hook,
};
