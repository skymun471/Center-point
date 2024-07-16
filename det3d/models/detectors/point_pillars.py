from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy

# print("=================================playground_0708================================")
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# print("=================================playground_0708================================")
@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds, _ = self.bbox_head(x)

        print("=================================playground_0708================================")
        # print(type(preds),len(preds), preds)
        # hm_tensors = [item['hm'] for item in preds if 'hm' in item]
        #
        # output_dir = '/home/milab20/PycharmProjects/Center_point/CenterPoint/hm_output'
        # for i, hm in enumerate(hm_tensors):
        #     file_path = os.path.join(output_dir, f'heatmap_{i}.png')
        #     save_tensor_as_image(hm, file_path)
        #     print(f'Saved heatmap {i} to {file_path}')

        print("=================================playground_0708================================")

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            out = self.bbox_head.predict(example, preds, self.test_cfg)
            # print(out)
            return out
        # 이걸로 하면 single_inference 안됨
        # return self.forward_two_stage(example, return_loss, **kwargs)
    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)
        # print("Two_stage",boxes)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None


def save_tensor_as_image(tensor, file_path):
    # 텐서를 NumPy 배열로 변환
    np_array = tensor.cpu().numpy()

    # 히트맵 이미지를 저장할 경로 생성 (디렉토리가 존재하지 않으면 생성)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 히트맵을 이미지로 저장
    plt.imshow(np_array[0, 0], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(file_path)
    plt.close()