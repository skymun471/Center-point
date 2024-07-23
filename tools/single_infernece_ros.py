import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
from visualization_msgs.msg import Marker, MarkerArray

# Import necessary modules from det3d
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

# Define colors for different labels
LABEL_COLORS = {
    0: (1.0, 0.0, 0.0),  # car - red
    1: (0.0, 1.0, 0.0),  # truck - green
    2: (0.0, 0.0, 1.0),  # construction vehicle - blue
    3: (1.0, 1.0, 0.0),  # bus - yellow
    4: (1.0, 0.0, 1.0),  # trailer - magenta
    5: (0.0, 1.0, 1.0),  # barrier - cyan
    6: (1.0, 0.5, 0.0),  # motorcycle - orange
    7: (0.5, 0.0, 0.5),  # bicycle - purple
    8: (0.5, 0.5, 0.5),  # pedestrian - grey
    9: (0.0, 0.5, 0.5)   # traffic cone - teal
}

def yaw2quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle to quaternion."""
    return Quaternion(axis=[0, 0, 1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    """Get indices of annotations based on threshold."""
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices

def remove_low_score_nu(image_anno, thresh):
    """Remove low score annotations."""
    img_filtered_annotations = {}
    # print(image_anno)
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()

    car_indices = get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices = get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices = get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices = get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices = get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices = get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices = get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices = get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices = get_annotations_indices(9, 0.1, label_preds_, scores_)

    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices +
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])
    return img_filtered_annotations

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None

    def initialize(self):
        """Initialize the processor."""
        self.read_config()
        self.print_model_structure()

    def read_config(self):
        """Read configuration file."""
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        # Create voxel generator
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def print_model_structure(self):
        """Print the structure of the model."""
        print("Model structure:")
        print(self.net)
        print("\nModel parameters:")
        for name, param in self.net.named_parameters():
            print(f"{name}: {param.size()}")

    def run(self, points):
        """Run the detection model on the input points."""
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 5
        self.points = points.reshape([-1, num_features])
        self.points[:, 4] = 0  # timestamp value

        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]

        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)
        # print(outputs)
        outputs = remove_low_score_nu(outputs, 0.9)
        print("outputs",outputs)
        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        print(f"  total cost time: {time.time() - t_t}")

        return scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=float):
    """Extract xyz points from point cloud array."""
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """Create a sensor_msgs.PointCloud2 from an array of points."""
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    """Callback function for LiDAR messages."""
    t_t = time.time()
    arr_bbox = BoundingBoxArray()
    marker_array = MarkerArray()

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    print("  ")
    scores, dt_box_lidar, types = proc_1.run(np_p)

    dets = dt_box_lidar[:, [0, 1, 2, 3, 4, 5, 6]]
    info_data = np.stack((types, scores), axis=1)
    # print("dt_box_lidar",dt_box_lidar)
    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            q = yaw2quaternion(float(dt_box_lidar[i][8]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]
            bbox.pose.position.x = float(dt_box_lidar[i][0])
            bbox.pose.position.y = float(dt_box_lidar[i][1])
            bbox.pose.position.z = float(dt_box_lidar[i][2])
            bbox.dimensions.x = float(dt_box_lidar[i][4])
            bbox.dimensions.y = float(dt_box_lidar[i][3])
            bbox.dimensions.z = float(dt_box_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = int(types[i])
            arr_bbox.boxes.append(bbox)

            # Marker for RViz
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = bbox.pose
            marker.scale.x = bbox.dimensions.x
            marker.scale.y = bbox.dimensions.y
            marker.scale.z = bbox.dimensions.z

            # Set marker color based on label
            color = LABEL_COLORS.get(bbox.label, (1.0, 1.0, 1.0))  # default color: white
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8

            marker_array.markers.append(marker)

            # Text Marker for label
            text_marker = Marker()
            text_marker.header.frame_id = msg.header.frame_id
            text_marker.header.stamp = rospy.Time.now()
            text_marker.id = i + 1000  # Ensure unique ID for text marker
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = bbox.pose
            text_marker.scale.z = 1.0  # Height of text
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"Label: {bbox.label}"
            marker_array.markers.append(text_marker)

    # print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = msg.header.frame_id
    arr_bbox.header.stamp = msg.header.stamp

    if len(arr_bbox.boxes) != 0:
        pub_arr_bbox.publish(arr_bbox)
        # print("Published arr_bbox")
    else:
        pub_arr_bbox.publish(arr_bbox)
        # print("Published empty arr_bbox")

    pub_marker_array.publish(marker_array)
    # print("Published marker_array")

if __name__ == "__main__":
    global proc
    ## CenterPoint
    config_path = 'configs_tools/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py'
    # model_path = '/home/milab20/PycharmProjects/Center_point/CenterPoint/work_dirs/pp_CenterPoint_pretrain/latest.pth'
    model_path= '/home/milab20/PycharmProjects/Center_point/CenterPoint/work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms/latest.pth'
    proc_1 = Processor_ROS(config_path, model_path)

    proc_1.initialize()

    rospy.init_node('centerpoint_ros_node')
    sub_lidar_topic = [
        "/velodyne_points",
        "/top/rslidar_points",
        "/points_raw",
        "/lidar_protector/merged_cloud",
        "/merged_cloud",
        "/lidar_top",
        "/roi_pclouds",
        "/kitti/velo/pointcloud"
    ]

    sub_ = rospy.Subscriber(sub_lidar_topic[7], PointCloud2, rslidar_callback, queue_size=1, buff_size=2 ** 24)

    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)
    pub_marker_array = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=1)

    print("[+] CenterPoint ros_node has started!")
    rospy.spin()
