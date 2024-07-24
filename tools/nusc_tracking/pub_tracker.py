import numpy as np
import copy
from track_utils import greedy_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import copy
import importlib
import sys

association_module_dir = '/home/milab20/PycharmProjects/Center_point/CenterPoint/tools/'
sys.path.append(association_module_dir)

from Associate import AB3DMOT  # 위 코드가 포함된 파일명을 ab3dmot.py로 가정

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
# NUSCENE_CLS_VELOCITY_ERROR = {
#   'car':4,
#   'truck':4,
#   'bus':5.5,
#   'trailer':3,
#   'pedestrian':1,
#   'motorcycle':13,
#   'bicycle':3,
# }

NUSCENE_CLS_VELOCITY_ERROR = {
    0: 4.0, 1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0,
    5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0
}

class PubTracker(object):
  def __init__(self,  hungarian=False, max_age=0):
    self.hungarian = hungarian
    self.max_age = max_age

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR
    self.tracker = AB3DMOT(ID_init=0)  # 트래커 초기화
    self.reset()

  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, data, time_lag):
    # 만약 results가 비어있다면, self.tracks도 비우고 빈 리스트를 반환
    if len(data) == 0:
      # print("dfsdfsfsdfsfsdfsdf")
      self.tracks = []
      return [], [], []

    else:

      temp = []

      results = []

      for item in data:
        box3d_lidar = item['box3d_lidar'].cpu().numpy()
        scores = item['scores'].cpu().numpy()
        label_preds = item['label_preds'].cpu().numpy()
        token = item['metadata']['token']

        for i in range(len(box3d_lidar)):
          result = {
            'sample_token': token,
            'translation': box3d_lidar[i][:3].tolist(),
            'velocity': box3d_lidar[i][3:5].tolist(),
            'size': box3d_lidar[i][5:8].tolist(),
            'rotation': box3d_lidar[i][8].tolist(),
            'detection_score': scores[i].item(),
            'label_preds': int(label_preds[i]),
            'detection_name': 'unknown',
            'attribute_name': 'unknown'  # Replace with the correct attribute if available
          }
          results.append(result)



      for det in results:
        # filter out classes not evaluated for tracking
        # NUSCENES_TRACKING_NAMES에 포함되지 않은 클래스는 필터링하여 무시
        # if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
        #   continue

        # translation의 처음 두 요소를 numpy 배열로 변환하여 'ct'에 저장 (중심 좌표)
        det['ct'] = np.array(det['translation'][:2])

        # velocity의 처음 두 요소를 numpy 배열로 변환하고, time_lag를 곱해 추적 오프셋 계산
        det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag

        # detection_name을 인덱스로 변환하여 'label_preds'에 저장
        # det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])

        # print("dsdsdsdsds",det['label_preds'])
        temp.append(det)

      # 유효한 detections만 포함된 리스트로 갱신
      results = temp


      # 데이터 추출 및 변환


    # print(results)
    # =======================================================================================================
    # print("위 조건에 의해 정제된 det 데이터 확인", results)
    '''
    results
    {'sample_token': '3950bd41f74548429c0f7700ff3d8269',
     'translation': [587.2733082842324, 1654.462978249677, 1.1674389887159642],
     'size': [1.942120909690857, 4.49296236038208, 1.6254323720932007],
     'rotation': [0.9696506293134629, 0.011323225182605565, 0.029827568305485344, -0.24240412086524069],
     'velocity': [7.360823584934132, -3.658063897239092],
     'detection_name': 'car',
     'detection_score': 0.7712615728378296,
     'attribute_name': 'vehicle.moving',
     'ct': array([587.27330828, 1654.46297825]),
     'tracking': array([-3.68361283, 1.83062275]),
     'label_preds': 2}
     '''
    # =======================================================================================================



    # ================================================JPDA================================================
    score_threshold = 0.3
    box3d_lidar_list = []
    scores_list = []
    label_preds_list = []

    for output in results:
      box3d_lidar = np.array([output["translation"] + [output["size"][1], output["size"][0], output["size"][2],
                                                       output["rotation"]]])
      scores = np.array([output["detection_score"]])
      label_preds = np.array([output["label_preds"]])

      mask = scores >= score_threshold
      box3d_lidar = box3d_lidar[mask]
      scores = scores[mask]
      label_preds = label_preds[mask]

      # box3d_lidar_list.append(box3d_lidar)
      # scores_list.append(scores)
      # label_preds_list.append(label_preds)

      if box3d_lidar.size > 0:
        box3d_lidar_list.append(box3d_lidar)
        scores_list.append(scores)
        label_preds_list.append(label_preds)

    if not box3d_lidar_list:
      return [], [], []

    box3d_lidar = np.concatenate(box3d_lidar_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    label_preds = np.concatenate(label_preds_list, axis=0)

    # dets format : hwlxyzo (height, width, length, x, y, z, orientation)
    dets = box3d_lidar[:, [5, 4, 3, 0, 1, 2, 6]]  # z, w, l, x, y, z, o
    # print("dets:", dets)
    info_data = np.stack((label_preds, scores), axis=1)

    dic_dets = {
      'dets': dets,
      'info': info_data
    }

    # JPDA를 사용하여 추적 수행
    jpda_results, affi, mat, un_mat, un_trk = self.tracker.track(dic_dets)

    # print("JPDA results",jpda_results)
    # print("mat:",   mat)
    # print("un_mat", un_mat)
    # print("un_trk", un_trk)
    # ================================================JPDA================================================

    # ===========================================Center_track(1)==================================+==========
    # 현재 프레임의 detection 수
    N = len(results)

    # 이전 프레임의 track 수
    # 처음에는 아무것도 없겠지
    M = len(self.tracks)

    # N X 2
    # N X 2 배열 생성
    if 'tracking' in results[0]:
      # tracking 정보를 포함하여 각 detection의 중심 좌표 갱신
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32)
       for det in results], np.float32)
    else:
      # tracking 정보가 없는 경우, 단순히 중심 좌표만 사용
      dets = np.array(
        [det['ct'] for det in results], np.float32)

    # 현재 프레임의 각 detection에 대한 클래스 레이블 인덱스 배열
    item_cat = np.array([item['label_preds'] for item in results], np.int32) # N
    # =======================================================================================================
    # print('현재 프레임의 detection 클래스 레이블 확인',item_cat)
    # 현재 프레임의 det 예측된 cls 값을 모아놓음
    '''
    item_cat
    [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 5
     1 5 5 1 1 5 5 0 0 0 0 3 3 3 0 0 0 0 0 0 0 3 3 0 3 0 3 0 0 3 3 0 0 0 3 3 0
     3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4]
    '''
    # =======================================================================================================

    # 이전 프레임의 각 track에 대한 클래스 레이블 인덱스 배열
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

    # 각 detection의 클래스별 최대 허용 거리 오차 배열
    # max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

    max_diff = np.array([NUSCENE_CLS_VELOCITY_ERROR[int(box['label_preds'])] for box in results], np.float32)

    # max_diff = np.array([NUSCENE_CLS_VELOCITY_ERROR[int(label)] for box in results for label in box['label_preds']],
                        # np.float32)
    # =======================================================================================================
    # print("각 detection의 클래스별 최대 허용 거리 오차 배열",max_diff)

    '''
    max_diff
    [ 4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.
      4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.   4.
      4.   4.   3.   5.5  3.   3.   5.5  5.5  5.5  5.5  3.   3.   3.   3.
     13.   3.  13.   3.  13.  13.   3.   3.  13.  13.   3.   3.   3.   3.
     13.  13.  13.  13.   3.  13.  13.  13.   3.   3.   1.   1.   1.   1.
      1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
      1.   1.   1.   1.   1.   1.   1.   1. ]
    '''
    # =======================================================================================================

    # 이전 프레임의 각 track의 중심 좌표 배열
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2
    # =======================================================================================================
    # print("이전 프레임의 각 track의 중심 좌표 배열",tracks)
    '''
    tracks
     [[1312.984   1037.0889 ]
      [1295.8119  1043.2124 ]
      [1331.355   1090.1265 ]
      [1285.767   1046.9586 ]
      [1321.405   1033.9313 ]
      [1287.8683  1039.8705 ]
      [1296.2847   999.95667]
      [1284.1102  1026.1285 ]
      [1276.9661  1033.6013 ]
      [1285.4583  1032.3774 ]
      [1289.9736  1042.3418 ]]
    '''
    # =======================================================================================================

    # 첫 프레임이 아닌 경우 (즉, 이전 프레임에 추적된 객체가 있는 경우)
    # NOT FIRST FRAME
    if len(tracks) > 0:
      # 현재 프레임의 각 detection과 이전 프레임의 각 track 사이의 거리 계산
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # absolute distance in meter

      # 거리와 클래스 레이블을 기준으로 유효하지 않은 매칭 마스크 생성
      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      # 유효하지 않은 매칭의 거리를 매우 큰 값으로 설정하여 무시되도록 함
      dist = dist  + invalid * 1e18

      # ==========================================================
      # print("dist 행렬 확인: ",dist.shape)
      # 행과 열의 개수가 Trk와 det의 개수
      '''
      dist 행렬 확인:  (133, 152)
      '''
      # ==========================================================

      if self.hungarian:
        # 거리가 매우 큰 값을 가지는 요소를 1e18로  제한
        dist[dist > 1e18] = 1e18
        # 항가리안 알고리즘을 사용하여 매칭 수행
        matched_indices = np.array(linear_assignment(copy.deepcopy(dist)))
        # 결과의 전치 (행과 열을 바꿔서) 행렬 형태로 변환
        matched_indices = matched_indices.transpose()
        # ==========================================================
        # print("matched_indices 행렬",matched_indices)
        # ==========================================================
      else:
        # 탐욕 알고리즘을 사용하여 매칭 수행
        matched_indices = greedy_assignment(copy.deepcopy(dist))
        # ==========================================================
        # print("matched_indices 행렬", matched_indices)
        '''
         [[  0  10]
         [  1   6]
         [  2   2]
         [  3   0]
         [  4  22]
         [  5   1]
         [  6   9]
         [  7  24]]
        '''
        # ==========================================================

    # 첫 몇 프레임 (이전 프레임에 추적된 객체가 없는 경우)
    # first few frame
    else:
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    # 매칭되지 않은 detections의 인덱스 리스트 생성
    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]
    # ==========================================================
    # print("unmatched_dets의 인덱스 리스트",unmatched_dets)
    '''
    unmatched_dets
     [39, 41,05, 108, 109, 111, 112, 114, 115, 116,
      124, 125, 127, 128, 129, 130]
    '''
    # ==========================================================

    # 매칭되지 않은 tracks의 인덱스 리스트 생성
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]


    if self.hungarian:
      matches = []
      for m in matched_indices:
        # 거리가 매우 큰 매칭은 무시하고 unmatched_dets에 추가
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      # 매칭 결과 배열로 변환
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    # print("matches", matches)

    # ===========================================Center_track(1)==================================+==========

    '''
    matches, unmatched_dets, unmatched_tracks의 값을 리스트 형태로 변환해서 넣어주면 된다.
    but 칼만필터를 어떻게 적용해야 할 것인가?
    '''
    # TODO: (1)
    # results리스트로 변환해야함

    J_matches, j_unmatched_dets, j_unmatched_tracks = mat, un_mat, un_trk
    j_matches = np.array(J_matches).reshape(-1,2)


    # print("matches",matches)
    # print("j_matches",j_matches)
    # print("unmatched_dets",unmatched_dets)
    # print("unmatched_tracks",unmatched_tracks)



    # 두 딕셔너리를 비교하여 다른 trk id의 수를 카운트
    # loss_count = 0
    # for det_id in matches_dict:
    #   if det_id in j_matches_dict and matches_dict[det_id] != j_matches_dict[det_id]:
    #     loss_count += 1
    #
    # print("loss_count", loss_count)

    ret = []
    for m in matches:
      track = results[m[0]]
      # print(track)
      # 기존 track의 tracking_id를 사용
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']
      # 새로 매칭된 track의 나이 및 활성 상태 갱신
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    for i in unmatched_dets:
      track = results[i]
      # print("unmatcehd_track", track)
      # 새로운 tracking_id 할당
      self.id_count += 1
      track['tracking_id'] = self.id_count
      # 새로운 track의 나이 및 활성 상태 초기화
      track['age'] = 1
      track['active'] =  1
      ret.append(track)

    # 매칭되지 않은 track들 중 나이가 max_age 이하인 것들은 여전히 저장하지만, 현재 프레임에서는 출력하지 않음
    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output
    # the object in current frame
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.max_age:
        # track의 나이 증가
        track['age'] += 1
        # 활성 상태를 0으로 설정
        track['active'] = 0
        ct = track['ct']

        # 지난 시간 동안의 이동 계산
        # movement in the last second
        if 'tracking' in track:
            offset = track['tracking'] * -1 # move forward
            track['ct'] = ct + offset
        ret.append(track)

    self.tracks = ret
    # print("ret",self.tracks)

    return ret, matches, j_matches
