from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker
from opendbc.can.parser import get_raw_value
from opendbc.car.hyundai.carcontroller import CarController, stalk_checksum

# real 0x3C1 frames, Ioniq 5 PE 10-26 rlog 00000007 (every lever change of the drive)
REAL = ['3e50300000000000', '5260304010000000', 'ce70304000000000', 'f880300000000000', 'de90304000000000',
        '97a0300000000000', '27b0300041000000', 'bbc0300001000000', '9fd0300000000000',
        # GV80 10-27 rlog 00000333 segment 1: idle -> left tap -> idle -> right tap -> idle
        '5900300000000000', '5a10304010000000', '3620300000000000', '8630300041000000', '8740300000000000']


@pytest.fixture
def packer():
  return CANPacker('hyundai_canfd_generated')


def make_cc(packer, frame, mode, auto=False, kick_side=0, kick_until=0):
  # every attribute stalk_blinkers touches, so the test exercises the real code path
  return SimpleNamespace(frame=frame, blinker_stalk=mode, stalk_auto=auto, stalk_kick_side=kick_side,
                         stalk_kick_until=kick_until, CAN=SimpleNamespace(ECAN=0), packer=packer,
                         stalk_counter=None, stalk_source_counter=None, stalk_values=None,
                         stalk_press_side=0, stalk_press_frame=0)


def make_cs(packer, idle):
  sigs = packer.dbc.name_to_msg['BLINKER_STALKS'].sigs
  return SimpleNamespace(blinker_stalks={k: get_raw_value(bytes.fromhex(idle), s) for k, s in sigs.items()})


def send(packer, cc, cs):
  out = CarController.stalk_blinkers(cc, cs)
  if not out:
    return None
  (addr, dat, bus), = out
  assert (addr, bus) == (0x3C1, 0)
  assert stalk_checksum(dat[1:]) == dat[0]
  return dat


@pytest.mark.parametrize('frame', REAL)
def test_checksum_matches_real_frames(frame):
  dat = bytes.fromhex(frame)
  assert stalk_checksum(dat[1:]) == dat[0]


def test_first_frame_carries_the_tap_edge(packer):
  # stock always sends BLINKER + BLINKER_TAP first (GV80 82.96 s, Ioniq 5 3372.9 s); without that edge the BCM
  # ignores the request, which is why replaying the steady form alone never worked.
  sigs = packer.dbc.name_to_msg['BLINKER_STALKS'].sigs
  cc, cs = make_cc(packer, 0, 1), make_cs(packer, 'f880300000000000')
  dat = send(packer, cc, cs)
  assert get_raw_value(dat, sigs['LEFT_BLINKER']) == 1
  assert get_raw_value(dat, sigs['LEFT_BLINKER_TAP']) == 1
  assert get_raw_value(dat, sigs['RIGHT_BLINKER']) == 0
  assert get_raw_value(dat, sigs['COUNTER_ALT']) == 9   # stock 8 -> ours 9
  assert dat.hex() == 'fb90304010000000'


def test_tap_clears_after_200ms_and_holds(packer):
  sigs = packer.dbc.name_to_msg['BLINKER_STALKS'].sigs
  cc, cs = make_cc(packer, 0, 1), make_cs(packer, 'f880300000000000')
  assert send(packer, cc, cs).hex() == 'fb90304010000000'
  cc.frame = 20
  dat = send(packer, cc, cs)
  assert get_raw_value(dat, sigs['LEFT_BLINKER']) == 1
  assert get_raw_value(dat, sigs['LEFT_BLINKER_TAP']) == 0
  assert get_raw_value(dat, sigs['COUNTER_ALT']) == 10
  assert dat.hex() == '08a0304000000000'


def test_momentary_matches_the_gv80_lever(packer):
  # GV80's lever is momentary: BLINKER + TAP for 0.35 s and then one idle release frame at 0.40 s.
  # Both tap (5a10304010000000) and release (3620300000000000) match the car's own 10-27 rlog 00000333.
  cc, cs = make_cc(packer, 0, 3), make_cs(packer, '5900300000000000')
  sent = []
  for frame in range(60):
    cc.frame = frame
    dat = send(packer, cc, cs)
    if dat is not None:
      sent.append((frame, dat.hex()))
  assert sent == [(0, '5a10304010000000'), (20, '5a10304010000000'), (40, '3620300000000000')]
  # Switching directly to right tap (mode 4) while the physical stalk stays at counter 0 reproduces frames 4 and 5
  cc.blinker_stalk = 4
  sent_right = []
  for frame in range(60, 120):
    cc.frame = frame
    dat = send(packer, cc, cs)
    if dat is not None:
      sent_right.append((frame, dat.hex()))
  assert sent_right == [(60, '8630300041000000'), (80, '8630300041000000'), (100, '8740300000000000')]


def test_momentary_right_uses_the_right_tap_bits(packer):
  sigs = packer.dbc.name_to_msg['BLINKER_STALKS'].sigs
  cc, cs = make_cc(packer, 0, 4), make_cs(packer, '5900300000000000')
  dat = send(packer, cc, cs)
  assert get_raw_value(dat, sigs['RIGHT_BLINKER']) == 1
  assert get_raw_value(dat, sigs['RIGHT_BLINKER_TAP']) == 1
  assert get_raw_value(dat, sigs['LEFT_BLINKER']) == 0


def test_counter_advances_across_direction_change_and_restart(packer):
  sigs = packer.dbc.name_to_msg['BLINKER_STALKS'].sigs
  cc, cs = make_cc(packer, 0, 1), make_cs(packer, 'e860300000000000')
  assert get_raw_value(send(packer, cc, cs), sigs['COUNTER_ALT']) == 7
  cc.frame = 20
  assert get_raw_value(send(packer, cc, cs), sigs['COUNTER_ALT']) == 8   # tap -> hold
  cc.frame = 40
  cc.blinker_stalk = 2                                                   # new press, same stock counter
  right = send(packer, cc, cs)
  assert get_raw_value(right, sigs['RIGHT_BLINKER']) == 1
  assert get_raw_value(right, sigs['RIGHT_BLINKER_TAP']) == 1
  assert get_raw_value(right, sigs['COUNTER_ALT']) == 9                  # keeps advancing, no repeat
  cc.frame = 60
  assert get_raw_value(send(packer, cc, cs), sigs['COUNTER_ALT']) == 10
  cc.blinker_stalk = 0
  assert CarController.stalk_blinkers(cc, cs) == []
  assert cc.stalk_counter is None
  cc.frame = 80
  cc.blinker_stalk = 1                                                   # fresh start resyncs from the car
  assert get_raw_value(send(packer, cc, cs), sigs['COUNTER_ALT']) == 7


def test_counter_wraps(packer):
  sigs = packer.dbc.name_to_msg['BLINKER_STALKS'].sigs
  cc, cs = make_cc(packer, 0, 1), make_cs(packer, 'f0f0300000000000')   # stock counter 15
  assert get_raw_value(send(packer, cc, cs), sigs['COUNTER_ALT']) == 0


def test_stalk_auto_kick_sends_one_gv80_momentary_tap_and_resets(packer):
  cc, cs = make_cc(packer, 0, 0, auto=True, kick_side=3, kick_until=100), make_cs(packer, '5900300000000000')
  sent = []
  for frame in range(110):
    cc.frame = frame
    dat = send(packer, cc, cs)
    if dat is not None:
      sent.append((frame, dat.hex()))
  assert sent == [(0, '5a10304010000000'), (20, '5a10304010000000'), (40, '3620300000000000')]
  assert cc.stalk_kick_side == 0 and cc.stalk_press_side == 0 and cc.stalk_counter is None
  # a subsequent kick while the physical stalk stays at counter 0 advances to counters 3 (tap) and 4 (release)
  cc.stalk_kick_side, cc.stalk_kick_until = 4, 210
  sent2 = []
  for frame in range(110, 220):
    cc.frame = frame
    dat = send(packer, cc, cs)
    if dat is not None:
      sent2.append((frame, dat.hex()))
  assert sent2 == [(110, '8630300041000000'), (130, '8630300041000000'), (150, '8740300000000000')]
