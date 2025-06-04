#!/usr/bin/env python
import copy
import functools
import os
import time
from pathlib import Path
from threading import Lock
from typing import List

import cv2
import numpy as np
import rospy
import torch as torch
from allegro_constants import AllegroConstants
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from pose_tracking.dataset.ds_common import adjust_img_for_torch
from pose_tracking.trainer_memotr import filter_by_score
from pose_tracking.utils.args_parsing import load_args_from_file, postprocess_args
from pose_tracking.utils.detr_utils import postprocess_detr_boxes
from pose_tracking.utils.geom import convert_2d_t_to_3d
from pose_tracking.utils.pipe_utils import get_model
from pose_tracking.utils.pose import convert_pose_vector_to_matrix
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from torchvision import transforms

from memotr.models.runtime_tracker import RuntimeTracker
from memotr.structures.track_instances import TrackInstances
from memotr.utils.nested_tensor import tensor_list_to_nested_tensor


# This class listens to 3 rgb camera feeds and converts incoming messages
# to opencv images.
class PoseTrackerNode:

    # IMG_SIZE = (320, 240)
    IMG_SIZE = (640, 480)

    def __init__(self):
        # Init node
        rospy.init_node("cam_feeds", anonymous=True)

        # Tick rate
        self.rate = rospy.Rate(15.0)

        # For safely writing to shared data.
        self.mutex = Lock()

        # Camera feed topics
        self.enabled_cameras = [0]
        self.cam_subs = []
        self.cam_imgs = [None] * len(self.enabled_cameras)
        for i in self.enabled_cameras:
            print(f"Subscribing to /cam_{i+1}/color/image_raw")
            self.cam_subs.append(rospy.Subscriber(f"/cam_{i+1}/color/image_raw", Image, self.getCamCallback(i)))

        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.encoding = "rgb8"

        # self.command_pose_sub = rospy.Subscriber("/allegro_vision/command_pose", Pose, self.getCommandPose)
        self._command_pose_tf = np.eye(4)

        self.cam_pose_pub = rospy.Publisher("/allegro_vision/pose", PoseStamped, queue_size=1)
        self.pose_msg = PoseStamped()
        self.pose_msg_copy = PoseStamped()

        self.display = False

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # pose tracker setup
        torch.set_grad_enabled(False)
        self.pose_to_mat_converter_fn = functools.partial(convert_pose_vector_to_matrix, rot_repr="rotation6d")

        artifact_dir = Path(__file__).parent / "artifacts"
        exp_name = "urgent_vicuna_6331"
        args_path = artifact_dir / exp_name / "args.yaml"
        self.args.ckpt_path = artifact_dir / exp_name / "model_best.pth"
        assert (
            self.args.ckpt_path.exists()
        ), f"Checkpoint path for experiment {exp_name} does not exist at {self.args.ckpt_path}."
        assert args_path.exists(), f"Args path for experiment {exp_name} does not exist at {args_path}."
        self.args = load_args_from_file(args_path)
        self.args = postprocess_args(self.args, use_if_provided=False)
        self.args.use_ddp = False
        self.args.num_classes = 1
        self.model = get_model(self.args, num_classes=self.args.num_classes)
        self.model.eval().cuda()
        self.tracker = RuntimeTracker(
            det_score_thresh=0.5,
            track_score_thresh=0.5,
            miss_tolerance=30,
            use_motion=False,
            motion_min_length=0,
            motion_max_length=0,
            visualize=False,
            use_dab=self.args.tf_use_dab,
            matcher=None,
        )
        self.tracks = [
            TrackInstances(
                hidden_dim=self.model.hidden_dim,
                num_classes=self.model.num_classes,
                use_dab=self.args.tf_use_dab,
            ).to("cuda")
        ]
        self.failed_ts = []
        self.pose_mat_pred_abs_last = None
        self.t = 0
        # ensure intrinsics matches IMG_SIZE
        self.intrinsics = torch.cat(
            [torch.tensor(AllegroConstants.CAMERA_INTRINSICS[f"Camera{i}"]).view(3, 3) for i in self.enabled_cameras],
            dim=1,
        )

    def getCamCallback(self, cam_idx):
        def getCam(image):
            self.mutex.acquire()
            self.cam_imgs[cam_idx] = CvBridge().imgmsg_to_cv2(image, desired_encoding=self.encoding)
            # save the image
            # t = rospy.Time.now()
            # t = 0 if not hasattr(self, "t") else self.t
            # today = time.strftime("%Y-%m-%d")
            # os.makedirs(f'data/{today}', exist_ok=True)
            # cv2.imwrite(f'data/{today}/cam_{cam_idx}_{t}.png', self.cam_imgs[cam_idx][:,:,::-1])
            # print(f'cam {cam_idx} img shape {self.cam_imgs[cam_idx].shape}')
            self.mutex.release()

        return getCam

    def getCameraData(self):
        return self.cam_imgs

    def run(self):

        get_pose_count = 0
        device = "cuda"
        clf_score_thresh = 0.5

        # Spin ros to pull in data.
        while not rospy.is_shutdown():
            images = self.getCameraData()
            if images[0] is None:
                continue

            images = [cv2.resize(img, self.IMG_SIZE, interpolation=cv2.INTER_LINEAR) for img in images]
            assert len(images) == 1, len(images)
            cam1_img = images[0]
            rgb = adjust_img_for_torch(cam1_img)[None]

            intrinsics = self.intrinsics
            if not isinstance(intrinsics, list):
                intrinsics = [intrinsics.float()]

            with torch.inference_mode():

                t1 = time.monotonic()

                frame = tensor_list_to_nested_tensor(tensor_list=rgb).to(device)
                res = self.model(frame=frame, tracks=self.tracks)
                previous_tracks, new_tracks = self.tracker.update(model_outputs=res, tracks=self.tracks)
                self.tracks: List[TrackInstances] = self.model.postprocess_single_frame(
                    previous_tracks, new_tracks, None
                )

                tracks_result = self.tracks[0].to(torch.device("cpu"))

                tracks_result = filter_by_score(tracks_result, thresh=clf_score_thresh)
                tracks_result.boxes = postprocess_detr_boxes(
                    tracks_result.boxes, target_sizes=torch.tensor([self.IMG_SIZE])
                )
                track_ts = tracks_result.ts
                track_rots = tracks_result.rots
                if len(track_ts) > 1:
                    # if there is a false positive, pick by the highest clf score or do nms
                    scores = torch.max(tracks_result.scores, dim=-1).values
                    det_res = {
                        "bbox": tracks_result.boxes,
                        "labels": tracks_result.labels,
                        "scores": scores,
                        "track_ids": tracks_result.ids,
                    }
                    print(f"{det_res=}")
                    track_ts = track_ts[torch.argmin(scores)][None]
                    track_rots = track_rots[torch.argmin(scores)][None]
                if len(track_ts) > 0:
                    if self.args.do_predict_2d_t:
                        center_depth_pred = track_ts[..., 2:]
                        t_pred_2d = track_ts[..., :2]
                        convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                            t_pred_2d,
                            center_depth_pred,
                            [i.cpu() for i in intrinsics],
                            hw=self.IMG_SIZE[::-1],
                        )
                        track_ts = convert_2d_t_pred_to_3d_res["t_pred"]
                    pose_mat_pred_abs = self.pose_to_mat_converter_fn(torch.cat([track_ts, track_rots], dim=-1))
                else:
                    pose_mat_pred_abs = self.pose_mat_pred_abs_last
                    self.failed_ts.append(self.t)

                self.t += 1

                rotm_cam, t_cam = (
                    pose_mat_pred_abs[..., :3, :3],
                    pose_mat_pred_abs[..., :3, 3],
                )
                self.pose_mat_pred_abs_last = pose_mat_pred_abs

                t2 = time.monotonic()
                print(f"model time {t2 - t1}")

                t1 = time.monotonic()

                T_cam_cube = torch.zeros(4, 4)
                T_cam_cube[:3, :3] = rotm_cam[0]
                T_cam_cube[:3, 3] = t_cam[0]
                T_cam_cube[3, 3] = 1.0

                T_w_cam = AllegroConstants.CAMERA_EXTRINSICS[f"Camera{self.enabled_cameras[0]}"]

                T_w_cube_pred = T_w_cam @ T_cam_cube.cpu().numpy()

                p = T_w_cube_pred[:3, 3]
                q = R.from_matrix(T_w_cube_pred[:3, :3]).as_quat()

                get_pose_count += 1

                self.pose_msg.pose.position.x = p[0]
                self.pose_msg.pose.position.y = p[1]
                self.pose_msg.pose.position.z = p[2]
                self.pose_msg.pose.orientation.x = q[0]
                self.pose_msg.pose.orientation.y = q[1]
                self.pose_msg.pose.orientation.z = q[2]
                self.pose_msg.pose.orientation.w = q[3]

                self.pose_msg.header.stamp = rospy.Time.now()
                self.pose_msg.header.frame_id = "origin"

                self.pose_msg_copy = copy.deepcopy(self.pose_msg)

                self.cam_pose_pub.publish(self.pose_msg_copy)

            self.rate.sleep()
            t2 = time.monotonic()


if __name__ == "__main__":
    # Object for getting cam data.

    camera_feeds = PoseTrackerNode()
    camera_feeds.run()
