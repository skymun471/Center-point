import numpy as np
import matplotlib.pyplot as plt
from box import Box3D  # Box3D 모듈이 포함된 파일을 정확히 지정해야 합니다.
from Associate import AB3DMOT  # 위 코드가 포함된 파일명을 ab3dmot.py로 가정


# 가상의 detection 데이터를 생성하는 함수
def generate_dummy_detections(seed):
    np.random.seed(seed)
    detections = {
        'dets': [Box3D(np.random.rand(7)) for _ in range(5)],
        'info': [np.random.rand(2) for _ in range(5)]
    }
    return detections


# 메인 테스트 함수
def main():
    tracker = AB3DMOT(ID_init=1)

    # 두 개의 프레임 데이터를 생성
    detections_frame_0 = generate_dummy_detections(seed=0)
    detections_frame_1 = generate_dummy_detections(seed=1)

    num_iterations = 10

    for i in range(num_iterations):
        if i % 2 == 0:
            detections = detections_frame_0
            frame_id = 0
        else:
            detections = detections_frame_1
            frame_id = 1

        tracks, new_ids = tracker.track(detections)
        print(f"Iteration {i + 1} - Frame {frame_id}")
        print("Tracks:", tracks)
        print("New IDs:", new_ids)
        print("=" * 30)

    # 트래킹 시간 그래프 저장
    tracker.save_tracking_time_graph()


if __name__ == '__main__':
    main()