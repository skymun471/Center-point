import datetime

import numpy as np
import copy
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import itertools
import matplotlib.pyplot as plt
from box import Box3D
import os
import multiprocessing

np.set_printoptions(suppress=True, precision=3)

class Filter(object):
    def __init__(self, bbox3D, info, ID):
        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1
        self.info = info

class KF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1], [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])
        self.kf.P[7:, 7:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.x[:7] = self.initial_pos.reshape((7, 1))

    def compute_innovation_matrix(self):
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

    def get_velocity(self):
        return self.kf.x[7:]

class AB3DMOT:
    def __init__(self, ID_init=1):
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.tracking_times = []
        self.id_past_output = []
        self.id_past = []
        self.id_now_output = []
        self.gate_threshold = 3
        self.max_age = 3
        self.min_hits = 2
        self.affi_process = True
        self.affi_save_dir = '/home/milab20/PycharmProjects/Center_point/CenterPoint/tools/affi_graphs'
    def track(self, dets_all):
        dets, info = dets_all['dets'], dets_all['info']  # dets: N x 7, float numpy array

        # print("dets in JPDA:", dets)
        self.frame_count += 1

        # Save the past outputs for ID correspondence during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        dets = self.process_dets(dets, info)

        trks = self.prediction()

        trk_innovation_matrix = None
        trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]

        self.thres = 0.1
        matched, unmatched_dets, unmatched_trks, affi = \
            self.jpda_data_association(dets, trks, None, None, None, trk_innovation_matrix)

        # print(f"matched : {matched}")
        # print(f"unmatched : {unmatched_dets}")
        # print(f"unmatched_trks : {unmatched_trks}")

        for track_idx in range(len(self.trackers)):
            if track_idx in matched:
                prob_sum = np.sum([affi[track_idx][det_idx] for det_idx in matched[track_idx]])

                if prob_sum > 0:
                    measurement_update = np.zeros_like(self.trackers[track_idx].kf.x[:7])

                    for det_idx in matched[track_idx]:
                        det_array = Box3D.bbox2array(dets[det_idx])  # 탐지를 배열로 변환
                        measurement = np.array(det_array[:7]).reshape(-1, 1)
                        measurement_update += affi[track_idx][det_idx] * measurement

                    measurement_update /= prob_sum

                    # 방향 보정 적용
                    for det_idx in matched[track_idx]:
                        det_array = Box3D.bbox2array(dets[det_idx])
                        trk_orientation, det_orientation = self.orientation_correction(
                            self.trackers[track_idx].kf.x[3], det_array[3]
                        )
                        self.trackers[track_idx].kf.x[3] = trk_orientation
                        measurement_update[3] = self.within_range(det_orientation)
                        break  # 첫 번째 매칭된 탐지의 방향을 사용

                    self.trackers[track_idx].kf.update(measurement_update)

                    # 매칭된 경우 초기화
                    self.trackers[track_idx].time_since_update = 0
                    self.trackers[track_idx].hits += 1
                else:
                    self.trackers[track_idx].time_since_update += 1
            else:
                self.trackers[track_idx].time_since_update += 1

        new_id_list = self.birth(dets, info, unmatched_dets)
        # print("new_id_list: ", new_id_list)
        # print(5)
        # Collect the results to return
        results = self.output()
        # print("results:", results)
        # print(6)
        if len(results) > 0:
            results = [np.concatenate(results)]  # h,w,l,x,y,z,theta, ID, other info, confidence
        else:
            results = [np.empty((0, 10))]

        self.id_now_output = results[0][:, 7].tolist()  # Only the active tracks that are outputted
        # print("id_now_output", self.id_now_output)
        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)
        # print(11)

        # matched_list = [(trk_idx, det_idx) for trk_idx, det_idxs in matched.items() for det_idx in det_idxs]
        matched_list = [[det_idx, trk_idx] for trk_idx, det_idxs in matched.items() for det_idx in det_idxs]
        sorted_matched_list = sorted(matched_list, key=lambda x: x[0])
        # self.save_affi_graph(affi, self.frame_count)
        # print("iter Tracking Finish!")
        # print("=====================================================================")
        return results, affi, sorted_matched_list, unmatched_dets, unmatched_trks

    def save_affi_graph(self, affi, frame_number):
        plt.figure(figsize=(10, 8))
        plt.imshow(affi, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Affinity Matrix at Frame {frame_number}')
        plt.xlabel('Detections')
        plt.ylabel('Tracks')
        plt.savefig(os.path.join(self.affi_save_dir, f'affi_frame_{frame_number:06d}.png'))
        plt.close()
    def output(self):
        num_trks = len(self.trackers)
        results = []

        # 각 트래커에 대해 역순으로 순회
        for trk in reversed(self.trackers):
            # 트래커의 상태 벡터에서 bbox 정보를 추출
            d = Box3D.array2bbox(trk.kf.x[:7].reshape((7,)))  # bbox location self
            d = Box3D.bbox2array_raw(d)

            # 조건 확인: 트래커가 유효한지 확인
            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                info_flat = np.array(trk.info).flatten()
                results.append(np.concatenate((d, [trk.id], info_flat)).reshape(1, -1))
            else:
                # print("Tracker does not meet the conditions.")
                pass
            num_trks -= 1

            # 소멸 조건 확인: 최대 업데이트 시간 초과
            if (trk.time_since_update >= self.max_age):
                # print("Removing tracker ID:", trk.id)
                self.trackers.pop(num_trks)

        # 디버깅 출력: 최종 결과
        # print("Final results:", results)

        return results
    def process_affi(self, affi, matched, unmatched_dets, new_id_list):

        trk_id = self.id_past  # ID in the trks for matching

        det_id = [-1 for _ in range(affi.shape[1])]  # initialization
        for track_idx, det_idxs in matched.items():
            if len(det_idxs) > 0:
                det_id[det_idxs[0]] = trk_id[track_idx]  # Use the first matched detection index

        count = 0
        assert len(unmatched_dets) == len(new_id_list), 'unmatched_dets와 new_id_list의 길이가 같아야 합니다.'


        for unmatch_tmp in unmatched_dets:
            det_id[unmatch_tmp] = new_id_list[count]  # new_id_list is in the same order as unmatched_dets
            count += 1
        assert not (-1 in det_id), 'error, still have invalid ID in the detection list'

        return affi

    def birth(self, dets, info, unmatched_dets):

        new_id_list = list()  # new ID generated for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            self.ID_count[0] += 1

        return new_id_list
    def prediction(self):

        trks = []
        for t in range(len(self.trackers)):
            # propagate locations
            kf_tmp = self.trackers[t]
            kf_tmp.kf.predict()
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])
            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks
    def within_range(self, theta):

        if theta >= np.pi: theta -= np.pi * 2
        if theta < -np.pi: theta += np.pi * 2

        return theta

    def diff_orientation_correction(self, diff):
        """
        return the angle diff = det - trk
        if angle diff > 90 or < -90, rotate trk and update the angle diff
        """
        if diff > np.pi / 2:  diff -= np.pi
        if diff < -np.pi / 2: diff += np.pi
        return diff

    def process_dets(self, dets, info):

        dets_new = []
        for i, det in enumerate(dets):
            det_tmp = Box3D.array2bbox_raw(det, info[i, :])
            dets_new.append(det_tmp)
        # dets_new = [Box3D.array2bbox_raw(det,info) for det in dets]
        return dets_new

    def m_distance(self, det, trk, trk_inv_innovation_matrix=None):
        det_array = Box3D.bbox2array(det)[:7]
        trk_array = Box3D.bbox2array(trk)[:7]
        diff = np.expand_dims(det_array - trk_array, axis=1)
        corrected_yaw_diff = self.diff_orientation_correction(diff[3])
        diff[3] = corrected_yaw_diff

        if trk_inv_innovation_matrix is not None:
            dist = np.sqrt(np.matmul(np.matmul(diff.T, trk_inv_innovation_matrix), diff)[0][0])
        else:
            dist = np.sqrt(np.dot(diff.T, diff))
        # print("dist:",dist)
        return dist

    def orientation_correction(self, theta_pre, theta_obs):

        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:
            theta_pre += np.pi
            theta_pre = self.within_range(theta_pre)

        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0:
                theta_pre += np.pi * 2
            else:
                theta_pre -= np.pi * 2

        return theta_pre, theta_obs
    # def jpda_data_association(self, dets, trks, metric, thres, algm, innovation_matrices):
    #     num_tracks = len(trks)
    #     num_detections = len(dets)
    #     print("num_tracks", num_tracks)
    #     print("num_detections", num_detections)
    #     association_probs = np.zeros((num_tracks, num_detections))
    #
    #     inv_innovation_matrices = [np.linalg.inv(matrix) for matrix in innovation_matrices]
    #
    #     for t_idx in range(num_tracks):
    #         for d_idx in range(num_detections):
    #             distance = self.m_distance(dets[d_idx], trks[t_idx], inv_innovation_matrices[t_idx])
    #             if distance < self.gate_threshold:
    #                 probability = np.exp(-0.5 * distance)
    #             else:
    #                 probability = 0.0
    #             association_probs[t_idx][d_idx] = probability
    #
    #     print("association_probs:", association_probs)
    #
    #     for t_idx in range(num_tracks):
    #         if np.sum(association_probs[t_idx]) > 0:
    #             association_probs[t_idx] /= np.sum(association_probs[t_idx])
    #     print("nor_association_probs:", association_probs)
    #
    #     log_association_probs = np.log(association_probs + 1e-9)
    #
    #     if num_detections >= num_tracks:
    #         valid_assignments = list(itertools.permutations(range(num_detections), num_tracks))
    #     else:
    #         valid_assignments = list(itertools.permutations(range(num_tracks), num_detections))
    #
    #     posterior_probs = np.zeros((num_tracks, num_detections))
    #     if num_detections >= num_tracks:
    #         for assignment in valid_assignments:
    #             log_prob = 0.0
    #             valid = True
    #             for t_idx in range(num_tracks):
    #                 if log_association_probs[t_idx][assignment[t_idx]] == -np.inf:
    #                     valid = False
    #                     break
    #                 log_prob += log_association_probs[t_idx][assignment[t_idx]]
    #             if valid:
    #                 prob = np.exp(log_prob)
    #                 for t_idx in range(num_tracks):
    #                     posterior_probs[t_idx][assignment[t_idx]] += prob
    #     else:
    #         for assignment in valid_assignments:
    #             log_prob = 0.0
    #             valid = True
    #             for d_idx in range(num_detections):
    #                 if log_association_probs[assignment[d_idx]][d_idx] == -np.inf:
    #                     valid = False
    #                     break
    #                 log_prob += log_association_probs[assignment[d_idx]][d_idx]
    #             if valid:
    #                 prob = np.exp(log_prob)
    #                 for d_idx in range(num_detections):
    #                     posterior_probs[assignment[d_idx]][d_idx] += prob
    #
    #     for t_idx in range(num_tracks):
    #         if np.sum(posterior_probs[t_idx]) > 0:
    #             posterior_probs[t_idx] /= np.sum(posterior_probs[t_idx])
    #     print("posterior_probs:", posterior_probs)
    #
    #     row_ind, col_ind = linear_sum_assignment(-posterior_probs)
    #
    #     matched_detections = set()
    #     matched = {}
    #     for t_idx, d_idx in zip(row_ind, col_ind):
    #         if association_probs[t_idx][d_idx] > 0:
    #             matched[t_idx] = [d_idx]
    #             matched_detections.add(d_idx)
    #
    #     unmatched_dets = [d_idx for d_idx in range(num_detections) if d_idx not in matched_detections]
    #     unmatched_trks = [t_idx for t_idx in range(num_tracks) if t_idx not in matched]
    #
    #     return matched, unmatched_dets, unmatched_trks, posterior_probs

    # def jpda_data_association(self, dets, trks, metric, thres, algm, innovation_matrices):
    #     num_tracks = len(trks)
    #     num_detections = len(dets)
    #     # print("num_tracks", num_tracks)
    #     # print("num_detections", num_detections)
    #     association_probs = np.zeros((num_tracks, num_detections))
    #
    #     inv_innovation_matrices = [np.linalg.inv(matrix) for matrix in innovation_matrices]
    #
    #     for t_idx in range(num_tracks):
    #         for d_idx in range(num_detections):
    #             distance = self.m_distance(dets[d_idx], trks[t_idx], inv_innovation_matrices[t_idx])
    #             if distance < self.gate_threshold:
    #                 probability = np.exp(-0.5 * distance)
    #             else:
    #                 probability = 0.0
    #             association_probs[t_idx][d_idx] = probability
    #
    #     # print("association_probs:", association_probs)
    #
    #     for t_idx in range(num_tracks):
    #         if np.sum(association_probs[t_idx]) > 0:
    #             association_probs[t_idx] /= np.sum(association_probs[t_idx])
    #     # print("nor_association_probs:", association_probs)
    #
    #     log_association_probs = np.log(association_probs + 1e-9)
    #
    #     # Use gating to reduce the number of valid assignments
    #     valid_assignments = []
    #     for t_idx in range(num_tracks):
    #         valid_dets = np.where(association_probs[t_idx] > 0)[0]
    #         valid_assignments.extend(itertools.product([t_idx], valid_dets))
    #
    #     posterior_probs = np.zeros((num_tracks, num_detections))
    #     for (t_idx, d_idx) in valid_assignments:
    #         if log_association_probs[t_idx][d_idx] != -np.inf:
    #             posterior_probs[t_idx][d_idx] += np.exp(log_association_probs[t_idx][d_idx])
    #
    #     for t_idx in range(num_tracks):
    #         if np.sum(posterior_probs[t_idx]) > 0:
    #             posterior_probs[t_idx] /= np.sum(posterior_probs[t_idx])
    #     # print("posterior_probs:", posterior_probs)
    #
    #     row_ind, col_ind = linear_sum_assignment(-posterior_probs)
    #
    #     matched_detections = set()
    #     matched = {}
    #     for t_idx, d_idx in zip(row_ind, col_ind):
    #         if association_probs[t_idx][d_idx] > 0:
    #             matched[t_idx] = [d_idx]
    #             matched_detections.add(d_idx)
    #
    #     unmatched_dets = [d_idx for d_idx in range(num_detections) if d_idx not in matched_detections]
    #     unmatched_trks = [t_idx for t_idx in range(num_tracks) if t_idx not in matched]
    #
    #     return matched, unmatched_dets, unmatched_trks, posterior_probs

    def jpda_data_association(self, dets, trks, metric, thres, algm, innovation_matrices):
        num_tracks = len(trks)
        num_detections = len(dets)
        # print("num_tracks", num_tracks)
        # print("num_detections", num_detections)
        association_probs = np.zeros((num_tracks, num_detections))

        inv_innovation_matrices = [np.linalg.inv(matrix) for matrix in innovation_matrices]

        for t_idx in range(num_tracks):
            for d_idx in range(num_detections):
                distance = self.m_distance(dets[d_idx], trks[t_idx], inv_innovation_matrices[t_idx])
                # print("distance",distance)
                if distance < self.gate_threshold:
                    probability = np.exp(-0.5 * distance)
                else:
                    probability = 0.0
                association_probs[t_idx][d_idx] = probability

        # print("association_probs:", association_probs)

        for t_idx in range(num_tracks):
            valid_prob_indices = association_probs[t_idx] > 0
            valid_probs_sum = np.sum(association_probs[t_idx][valid_prob_indices])
            if valid_probs_sum > 0:
                association_probs[t_idx][valid_prob_indices] /= valid_probs_sum
        # print("nor_association_probs:", association_probs)

        log_association_probs = np.log(association_probs + 1e-9)

        valid_assignments = []
        for t_idx in range(num_tracks):
            valid_dets = np.where(association_probs[t_idx] > 0)[0]
            valid_assignments.extend(itertools.product([t_idx], valid_dets))

        posterior_probs = np.zeros((num_tracks, num_detections))
        for (t_idx, d_idx) in valid_assignments:
            if log_association_probs[t_idx][d_idx] != -np.inf:
                posterior_probs[t_idx][d_idx] += np.exp(log_association_probs[t_idx][d_idx])

        for t_idx in range(num_tracks):
            valid_prob_indices = posterior_probs[t_idx] > 0
            valid_probs_sum = np.sum(posterior_probs[t_idx][valid_prob_indices])
            if valid_probs_sum > 0:
                posterior_probs[t_idx][valid_prob_indices] /= valid_probs_sum
        # print("posterior_probs:", posterior_probs)

        row_ind, col_ind = linear_sum_assignment(-posterior_probs)

        matched_detections = set()
        matched = {}
        for t_idx, d_idx in zip(row_ind, col_ind):
            if association_probs[t_idx][d_idx] > 0:
                matched[t_idx] = [d_idx]
                matched_detections.add(d_idx)

        unmatched_dets = [d_idx for d_idx in range(num_detections) if d_idx not in matched_detections]
        unmatched_trks = [t_idx for t_idx in range(num_tracks) if t_idx not in matched]

        return matched, unmatched_dets, unmatched_trks, posterior_probs