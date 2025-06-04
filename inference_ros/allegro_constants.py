import numpy as np
import glob
import yaml


class AllegroConstants:

    def load_origin_cam_tfs(yaml_fname: str):
        with open(yaml_fname, 'r') as stream:
            data = yaml.safe_load(stream)

        assert len(data) == 3, "There should be 3 cameras in the yaml file"

        T_o_c0 = np.array(data["T_origin_cam_1"])
        T_o_c1 = np.array(data["T_origin_cam_2"])
        T_o_c2 = np.array(data["T_origin_cam_3"])

        return T_o_c0, T_o_c1, T_o_c2

    T_origin_cam0, T_origin_cam1, T_origin_cam2 = load_origin_cam_tfs('cam_global_extrinsic_tf.yaml')

    # Relative to kuka base
    CAMERA_EXTRINSICS = {
        "Camera0": T_origin_cam0,
        "Camera1": T_origin_cam1,
        "Camera2": T_origin_cam2,
    }

    IMG_SIZE = (640, 480)
    CAMERA_INTRINSICS = {
        # Divide intrinsics by 2 to correct for the downsampling
        "Camera0": np.array([615.82031, 0.0, 324.87790, 0.0, 615.93530, 242.42068, 0.0, 0.0, 1.0]).reshape(
            (3, 3)
        ),
        "Camera1": np.array([621.26001, 0.0, 308.33221, 0.0, 620.72198, 231.11542, 0.0, 0.0, 1.0]).reshape(
            (3, 3)
        ),
        "Camera2": np.array([614.54767, 0.0, 322.00284, 0.0, 614.68542, 237.16765, 0.0, 0.0, 1.0]).reshape(
            (3, 3)
        ),
    }

    # T_c0_c1 reads as c1 wrt c0
    # T_c1_c0 reads as c0 wrt c1

    T_c0_c0 = np.eye(4)
    # T_c0_c1 = T_cam0_cam1
    # T_c0_c2 = T_cam0_cam2
    T_c0_c1 = np.matmul(
            np.linalg.inv(CAMERA_EXTRINSICS['Camera0']), CAMERA_EXTRINSICS['Camera1']
    )
    T_c0_c2 = np.matmul(
        np.linalg.inv(CAMERA_EXTRINSICS['Camera0']), CAMERA_EXTRINSICS['Camera2']
    )

    # Add KRt matrices of size (3, 4) with respect to camera 0.

    KRt = {
        "Camera0": np.matmul(
            CAMERA_INTRINSICS["Camera0"], np.linalg.inv(T_c0_c0)[0:3, 0:4]
        ),
        "Camera1": np.matmul(
            CAMERA_INTRINSICS["Camera1"], np.linalg.inv(T_c0_c1)[0:3, 0:4]
        ),
        "Camera2": np.matmul(
            CAMERA_INTRINSICS["Camera2"], np.linalg.inv(T_c0_c2)[0:3, 0:4]
        )

    }

    identity_pose = np.array([0, 0, 0] + [0.0, 0.0, 0.0, 1.0])