#!/usr/bin/env python3
import math
from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.gm.values import DBC, CanBus
from opendbc.car.interfaces import RadarInterfaceBase

RADAR_HEADER_MSG = 1120  # F_LRR_Obj_Header
SLOT_1_MSG = RADAR_HEADER_MSG + 1
NUM_SLOTS = 20

# Actually it's 0x47f, but can parser only reports
# messages that are present in DBC
LAST_RADAR_MSG = RADAR_HEADER_MSG + NUM_SLOTS


def create_radar_can_parser(car_fingerprint):
  # C1A-ARS3-A by Continental
  radar_targets = list(range(SLOT_1_MSG, SLOT_1_MSG + NUM_SLOTS))
  signals = list(zip(['FLRRNumValidTargets',
                      'FLRRSnsrBlckd', 'FLRRYawRtPlsblityFlt',
                      'FLRRHWFltPrsntInt', 'FLRRAntTngFltPrsnt',
                      'FLRRAlgnFltPrsnt', 'FLRRSnstvFltPrsntInt'] +
                     ['TrkRange'] * NUM_SLOTS + ['TrkRangeRate'] * NUM_SLOTS +
                     ['TrkRangeAccel'] * NUM_SLOTS + ['TrkAzimuth'] * NUM_SLOTS +
                     ['TrkWidth'] * NUM_SLOTS + ['TrkObjectID'] * NUM_SLOTS,
                     [RADAR_HEADER_MSG] * 7 + radar_targets * 6, strict=True))

  messages = list({(addr, 14) for (_, addr) in signals})

  return CANParser(DBC[car_fingerprint][Bus.radar], messages, CanBus.OBSTACLE)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)

    # CP.radarUnavailable == True 인 차량은 레이더 완전 비사용 (비전-only 모드)
    self.rcp = None if CP.radarUnavailable else create_radar_can_parser(CP.carFingerprint)

    # 한 프레임이 완성되었다고 보는 트리거 메시지
    self.trigger_msg = LAST_RADAR_MSG
    self.updated_messages = set()

    # Kans
    self.track_id = 0 # RadarPoint.trackId 생성용 내부 카운터
    self.prev_radar_fault = False # 이전 레이더 Fault 상태 기억

  def reset_radar(self):
    # Fault → 정상 복구 시 기존 타겟/트랙 날려줌
    self.pts.clear()
    self.track_id = 0
    self.updated_messages.clear()

  def update(self, can_strings):
    # 레이더가 완전히 비활성(CP.radarUnavailable=True)인 경우
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    # 아직 모든 target 메시지를 받지 못함
    if self.trigger_msg not in self.updated_messages:
      return None

    ret = structs.RadarData()
    header = self.rcp.vl[RADAR_HEADER_MSG]
    # CAN에러 먼저 처리
    if not self.rcp.can_valid:
      ret.errors.canError = True
      self.reset_radar()
      self.updated_messages.clear()
      return ret
    # Kans: Fault 검출
    fault = header['FLRRSnsrBlckd'] or header['FLRRSnstvFltPrsntInt'] or \
      header['FLRRYawRtPlsblityFlt'] or header['FLRRHWFltPrsntInt'] or \
      header['FLRRAntTngFltPrsnt'] or header['FLRRAlgnFltPrsnt']
    if fault:
      ret.errors.radarFault = True

    # Fault → 정상 복구 시 트랙 리셋
    if self.prev_radar_fault and not fault:
      self.reset_radar()
    self.prev_radar_fault = fault

    currentTargets = set()
    num_targets = header['FLRRNumValidTargets']
    if num_targets == 0:
      # 타겟 없으면 기존 트랙도 정리하고 빈 결과 리턴
      self.pts.clear()
      ret.points = []
      self.updated_messages.clear()
      return ret
    # Not all radar messages describe targets,
    # no need to monitor all of the self.rcp.msgs_upd
    for ii in self.updated_messages:
      if ii == RADAR_HEADER_MSG:
        continue

      cpt = self.rcp.vl[ii]
      # Zero distance means it's an empty target slot
      if cpt['TrkRange'] > 0.0:
        targetId = cpt['TrkObjectID']
        currentTargets.add(targetId)

        # 새로운 타겟이면 RadarPoint 생성 + 내부 track_id 부여
        if targetId not in self.pts:
          self.pts[targetId] = structs.RadarData.RadarPoint()
          self.pts[targetId].trackId = self.track_id
          self.track_id += 1

        distance = cpt['TrkRange']

        # 값 업데이트
        self.pts[targetId].dRel = distance  # from front of car
        # From driver's pov, left is positive
        self.pts[targetId].yRel = math.sin(cpt['TrkAzimuth'] * CV.DEG_TO_RAD) * distance
        self.pts[targetId].vRel = cpt['TrkRangeRate']
        self.pts[targetId].vLead = self.pts[targetId].vRel + self.v_ego
        self.pts[targetId].aRel = cpt['TrkRangeAccel'] # float('nan')
        self.pts[targetId].yvRel = 0  # float('nan') 도 가능하지만 0으로 고정
        self.pts[targetId].measured = True

    # 이전 프레임에서 사라진 타겟 제거
    for oldTarget in list(self.pts.keys()):
      if oldTarget not in currentTargets:
        del self.pts[oldTarget]

    ret.points = list(self.pts.values())
    self.updated_messages.clear()
    return ret