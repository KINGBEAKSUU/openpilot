"""NewUi 3 robot face: beat tracking on microphone audio (micd rawAudioData, 16 kHz int16)."""
import numpy as np
import pytest

from openpilot.selfdrive.ui.mici.onroad.robot_face_music import RATE, BeatTracker


def drums(bpm, secs, noise=0.0, seed=1):
  """Kick on 1 and 3, snare on 2 and 4, eighth-note hi-hats (and an off-beat kick every other bar)."""
  rng = np.random.default_rng(seed)
  n = int(secs * RATE)
  x = np.zeros(n)
  step = 60 / bpm / 4
  for s in range(int(secs / step)):
    i, k = int(s * step * RATE), s % 16

    def add(sig, i=i):
      j = min(n, i + sig.size)
      x[i:j] += sig[:j - i]
    if k in (0, 8) or (k == 10 and (s // 16) % 2):
      tt = np.arange(int(0.3 * RATE)) / RATE
      add(0.9 * np.sin(2 * np.pi * np.cumsum(140 * np.exp(-tt * 9) + 42) / RATE) * np.exp(-tt * 10))
    if k in (4, 12):
      tt = np.arange(int(0.2 * RATE)) / RATE
      add(0.4 * rng.standard_normal(tt.size) * np.exp(-tt * 18))
    if k % 2 == 0:
      tt = np.arange(int(0.05 * RATE)) / RATE
      add(0.08 * rng.standard_normal(tt.size) * np.exp(-tt * 60))
  x += noise * rng.standard_normal(n)
  return (np.clip(x, -1, 1) * 16000).astype(np.int16)


def road(secs, seed):
  """Engine hum, hiss and bumps at random moments: no beat."""
  rng = np.random.default_rng(seed)
  n = int(secs * RATE)
  t = np.arange(n) / RATE
  x = 0.3 * np.sin(2 * np.pi * np.cumsum(30 + 5 * np.sin(t * 0.3)) / RATE) + 0.15 * rng.standard_normal(n)
  for b in rng.uniform(0, secs, int(secs / 1.3)):
    i = int(b * RATE)
    tt = np.arange(int(0.1 * RATE)) / RATE
    j = min(n, i + tt.size)
    x[i:j] += (0.6 * np.exp(-tt * 30) * rng.standard_normal(tt.size))[:j - i]
  return (np.clip(x, -1, 1) * 16000).astype(np.int16)


def listen(audio):
  """Feed it like the UI does: 20 times a second, the clock first, then the new 50 ms chunk."""
  bt, beats, locked = BeatTracker(), [], []
  for c in range(audio.size // 800):
    now = (c + 1) * 0.05
    beats += [now] * len(bt.advance(0.05))
    bt.feed(audio[c * 800:(c + 1) * 800])
    locked.append(bt.locked)
  return bt, np.array(beats), np.array(locked)


@pytest.mark.parametrize('bpm', [90, 120, 150])
def test_finds_the_tempo_and_lands_on_the_beat(bpm):
  bt, beats, locked = listen(drums(bpm, 16, noise=0.05))
  assert locked[int(8 / 0.05):].all()                     # locked within 8 s and stays locked
  assert bt.bpm == pytest.approx(bpm, rel=0.02)
  period = 60 / bpm
  late = beats[beats > 9]
  err = (late + period / 2) % period - period / 2         # distance to the nearest beat of the music
  assert np.median(np.abs(err)) < 0.06                    # within one 50 ms UI frame or so
  assert len(late) == pytest.approx((16 - 9) / period, abs=2)


@pytest.mark.parametrize('seed', range(12))
def test_road_noise_never_makes_it_dance(seed):
  _, beats, locked = listen(road(30, seed))
  assert not locked.any() and beats.size == 0


def test_silence_and_music_stopping():
  bt, _, locked = listen(np.zeros(10 * RATE, np.int16))
  assert not locked.any()
  bt, _, _ = listen(np.concatenate((drums(120, 12), np.zeros(8 * RATE, np.int16))))
  assert not bt.locked                                    # the music stopped: it lets go
