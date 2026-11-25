# LFA (0x12A) camera-SCC forwarding queue: the standing-backlog trim in safety_hyundai_canfd.h.
# Each stock LFA frame from the camera carries the oldest queued host command. A host that runs
# slightly faster than the camera used to leave the queue 5-6 deep for good (54-64 ms late torque).
import pytest

from opendbc.safety.tests.test_hyundai_canfd_cluster import Hooks, native  # noqa: F401  (native: session fixture)

LFA = 0x12A
PARAM = 29  # camera SCC + HDA2, as test_hyundai_canfd_fwd_timer
T = 10000   # stock LFA period, us


def lfa(marker, torque=0, req=True):
  d = bytearray(16)
  raw = (torque + 1024) & 0x7FF
  d[5] = (raw & 0x7F) << 1
  d[6] = ((raw >> 7) & 0xF) | (0x10 if req else 0)
  d[10:14] = marker.to_bytes(4, "little")
  return bytes(d)


def marker(out):
  return int.from_bytes(out[1][10:14], "little")


def torque(out):
  return (((out[1][6] & 0xF) << 7) | (out[1][5] >> 1)) - 1024


def req(out):
  return bool(out[1][6] & 0x10)


class Stream:
  def __init__(self, native):
    self.h = Hooks(native, PARAM)
    self.now = 1000
    self.n = 0

  def push(self, tq=0, rq=True):
    self.n += 1
    assert self.h.tx(LFA, lfa(self.n, tq, rq), self.now, bus=0) == 3
    self.now += 50
    return self.n

  def stock(self):
    self.now += T
    return self.h.rx(LFA, bytes(16), self.now, bus=2)


def backlog(s, pops, depth=3, tq=0):
  # keep `depth` commands queued before every stock frame
  for _ in range(depth):
    s.push(tq)
  out = []
  for _ in range(pops):
    out.append(s.stock())
    s.push(tq)
  return out


def test_standing_backlog_is_trimmed_to_target(native):
  s = Stream(native)
  out = backlog(s, 400, depth=5)
  # 5 queued -> trimmed one at a time down to 2 before a pop: newest output lags the newest push by 1
  assert s.n - marker(out[-1]) == 2
  assert all(marker(b) - marker(a) in (1, 2) for a, b in zip(out, out[1:]))


def test_target_depth_is_left_alone(native):
  s = Stream(native)
  out = backlog(s, 400, depth=2)
  assert all(marker(b) - marker(a) == 1 for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_burst_is_sent_in_order(native, n):
  s = Stream(native)
  s.push(); s.push()
  first = [marker(s.stock()), marker(s.stock())]
  burst = [s.push() for _ in range(n)]
  seen = [marker(s.stock()) for _ in range(n)]
  assert first == [1, 2] and seen == burst


def _arm_with(s, tail):
  # Keep 3 commands queued before each stock frame. Commands c1..c3 are queued first; stock frame
  # i sends c_i and is followed by c_(i+3). Frames 1..99 are deep pops, so frame 100 is the first
  # that may trim, and it sees c100, c101, c102 queued with c99 as the last output.
  # `tail` = (torque, request) for c100, c101, c102; everything before is (0, request on).
  plan = [(0, True)] * 99 + list(tail)
  for k in range(3):
    s.push(*plan[k])
  outs = []
  for i in range(1, 100):
    outs.append(s.stock())
    s.push(*plan[i + 2])
  assert marker(outs[-1]) == 99 and s.n == 102
  return outs


def test_trim_happens_when_safe(native):
  s = Stream(native)
  _arm_with(s, [(0, True)] * 3)
  assert marker(s.stock()) == 101  # c100 trimmed


def test_no_trim_across_a_large_torque_step(native):
  s = Stream(native)
  _arm_with(s, [(10, True), (20, True), (30, True)])
  out = s.stock()
  assert marker(out) == 100 and torque(out) == 10  # skipping c100 would step 0 -> 20


@pytest.mark.parametrize("pattern", [(False, False, True), (False, True, True), (True, False, True)])
def test_request_off_frames_are_never_trimmed(native, pattern):
  s = Stream(native)
  _arm_with(s, [(0, rq) for rq in pattern])
  sent = [marker(s.stock()) for _ in range(3)]
  assert sent == [100, 101, 102]


def test_overflow_restarts_the_trim_count(native):
  s = Stream(native)
  _arm_with(s, [(0, True)] * 3)
  for _ in range(4):  # c103..c106: the 6-entry queue drops c100
    s.push()
  assert marker(s.stock()) == 101  # no extra trim on top of the overflow
