LM_OBJ_NAME_TO_ID = {
    "ape": 1,
    "benchvise": 2,
    "bowl": 3,
    "cam": 4,
    "can": 5,
    "cat": 6,
    "cup": 7,
    "driller": 8,
    "duck": 9,
    "eggbox": 10,
    "glue": 11,
    "holepuncher": 12,
    "iron": 13,
    "lamp": 14,
    "phone": 15,
}
LM_OBJ_ID_TO_NAME = {v: k for k, v in LM_OBJ_NAME_TO_ID.items()}
LM_SYMMETRY_OBJ_NAMES = {"eggbox", "glue", "cup", "bowl"}


YCBV_OBJ_NAME_TO_ID = {
    "master_chef_can": 1,
    "cracker_box": 2,
    "sugar_box": 3,
    "tomato_soup_can": 4,
    "mustard_bottle": 5,
    "tuna_fish_can": 6,
    "pudding_box": 7,
    "gelatin_box": 8,
    "potted_meat_can": 9,
    "banana": 10,
    "pitcher_base": 11,
    "bleach_cleanser": 12,
    "bowl": 13,
    "mug": 14,
    "power_drill": 15,
    "wood_block": 16,
    "scissors": 17,
    "large_marker": 18,
    "large_clamp": 19,
    "extra_large_clamp": 20,
    "foam_brick": 21,
}
YCBV_OBJ_ID_TO_NAME = {v: k for k, v in YCBV_OBJ_NAME_TO_ID.items()}
YCBV_SYMMETRY_OBJ_NAMES = {"bowl", "wood_block", "large_clamp", "extra_large_clamp", "foam_brick"}

TLESS_OBJ_NAME_TO_ID = {
    "obj01": 1,
    "obj02": 2,
    "obj03": 3,
    "obj04": 4,
    "obj05": 5,
    "obj06": 6,
    "obj07": 7,
    "obj08": 8,
    "obj09": 9,
    "obj10": 10,
    "obj11": 11,
    "obj12": 12,
    "obj13": 13,
    "obj14": 14,
    "obj15": 15,
    "obj16": 16,
    "obj17": 17,
    "obj18": 18,
    "obj19": 19,
    "obj20": 20,
    "obj21": 21,
    "obj22": 22,
    "obj23": 23,
    "obj24": 24,
    "obj25": 25,
    "obj26": 26,
    "obj27": 27,
    "obj28": 28,
    "obj29": 29,
    "obj30": 30,
}
TL_OBJ_ID_TO_NAME = {v: k for k, v in TLESS_OBJ_NAME_TO_ID.items()}
TLESS_SYMMETRY_OBJ_NAMES = ["obj{:02d}".format(obj_id + 1) for obj_id in range(0, 30)]

TUDL_OBJ_NAME_TO_ID = {"obj01": 1, "obj02": 2, "obj03": 3}
TU_OBJ_ID_TO_NAME = {v: k for k, v in TUDL_OBJ_NAME_TO_ID.items()}
TUDL_SYMMETRY_OBJ_NAMES = []

HB_OBJ_NAME_TO_ID = {
    "obj01": 1,
    "obj02": 2,
    "obj03": 3,
    "obj04": 4,
    "obj05": 5,
    "obj06": 6,
    "obj07": 7,
    "obj08": 8,
    "obj09": 9,
    "obj10": 10,
    "obj11": 11,
    "obj12": 12,
    "obj13": 13,
    "obj14": 14,
    "obj15": 15,
    "obj16": 16,
    "obj17": 17,
    "obj18": 18,
    "obj19": 19,
    "obj20": 20,
    "obj21": 21,
    "obj22": 22,
    "obj23": 23,
    "obj24": 24,
    "obj25": 25,
    "obj26": 26,
    "obj27": 27,
    "obj28": 28,
    "obj29": 29,
    "obj30": 30,
    "obj31": 31,
    "obj32": 32,
    "obj33": 33,
}
HB_OBJ_ID_TO_NAME = {v: k for k, v in HB_OBJ_NAME_TO_ID.items()}
HB_SYMMETRY_OBJ_NAMES = ["obj10", "obj11", "obj14"]

ICBIN_OBJ_NAME_TO_ID = {"obj01": 1, "obj02": 2}
ICBIN_OBJ_ID_TO_NAME = {v: k for k, v in ICBIN_OBJ_NAME_TO_ID.items()}
ICBIN_SYMMETRY_OBJ_NAMES = ["obj01"]

ITODD_OBJ_NAME_TO_ID = {
    "obj01": 1,
    "obj02": 2,
    "obj03": 3,
    "obj04": 4,
    "obj05": 5,
    "obj06": 6,
    "obj07": 7,
    "obj08": 8,
    "obj09": 9,
    "obj10": 10,
    "obj11": 11,
    "obj12": 12,
    "obj13": 13,
    "obj14": 14,
    "obj15": 15,
    "obj16": 16,
    "obj17": 17,
    "obj18": 18,
    "obj19": 19,
    "obj20": 20,
    "obj21": 21,
    "obj22": 22,
    "obj23": 23,
    "obj24": 24,
    "obj25": 25,
    "obj26": 26,
    "obj27": 27,
    "obj28": 28,
}
ITO_OBJ_ID_TO_NAME = {v: k for k, v in ITODD_OBJ_NAME_TO_ID.items()}
ITODD_SYMMETRY_OBJ_NAMES = [
    "obj02",
    "obj03",
    "obj04",
    "obj05",
    "obj07",
    "obj08",
    "obj09",
    "obj11",
    "obj12",
    "obj14",
    "obj17",
    "obj18",
    "obj19",
    "obj23",
    "obj24",
    "obj25",
    "obj27",
    "obj28",
]

YCBINEOAT_VIDEONAME_TO_OBJ = {
    "bleach0": "021_bleach_cleanser",
    "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
    "cracker_box_reorient": "003_cracker_box",
    "cracker_box_yalehand0": "003_cracker_box",
    "mustard0": "006_mustard_bottle",
    "mustard_easy_00_02": "006_mustard_bottle",
    "sugar_box1": "004_sugar_box",
    "sugar_box_yalehand0": "004_sugar_box",
    "tomato_soup_can_yalehand0": "005_tomato_soup_can",
}
YCBINEOAT_OBJ_TO_VIDEONAME = {
    "021_bleach_cleanser": "bleach0",
}
YCBINEOAT_SYMMETRY_OBJ_NAMES = []


COLOR_MAP = {
    "OrangeRed": [255, 69, 0],
    "LawnGreen": [124, 252, 0],
    "Cyan2": [0, 238, 238],
    "Yellow2": [238, 238, 0],
    "BlueViolet_Custom": [155, 48, 255],
    "Blue2": [0, 0, 238],
    "Magenta_Custom": [255, 131, 250],
    "DarkKhaki": [189, 183, 107],
    "Brown": [165, 42, 42],
    "Green2": [0, 234, 0],
    "Red2": [234, 0, 0],
    "DarkOrange": [255, 140, 0],
    "Gold": [255, 215, 0],
    "Purple_Custom": [160, 32, 240],
    "Yellow": [255, 255, 0],
    "Magenta": [255, 0, 255],
    "Cyan": [0, 255, 255],
    "LightGreen": [144, 238, 144],
    "LightBlue": [173, 216, 230],
    "LightCoral": [240, 128, 128],
    "LightPink": [255, 182, 193],
}
YCBV_OBJ_NAME_TO_COLOR = {
    "master_chef_can": COLOR_MAP["OrangeRed"],
    "cracker_box": COLOR_MAP["LawnGreen"],
    "sugar_box": COLOR_MAP["Cyan2"],
    "tomato_soup_can": COLOR_MAP["Yellow2"],
    "mustard_bottle": COLOR_MAP["BlueViolet_Custom"],
    "tuna_fish_can": COLOR_MAP["Blue2"],
    "pudding_box": COLOR_MAP["Magenta_Custom"],
    "gelatin_box": COLOR_MAP["DarkKhaki"],
    "potted_meat_can": COLOR_MAP["Brown"],
    "banana": COLOR_MAP["Green2"],
    "pitcher_base": COLOR_MAP["Red2"],
    "bleach_cleanser": COLOR_MAP["DarkOrange"],
    "bowl": COLOR_MAP["Gold"],
    "mug": COLOR_MAP["Purple_Custom"],
    "power_drill": COLOR_MAP["Yellow"],
    "wood_block": COLOR_MAP["Magenta"],
    "scissors": COLOR_MAP["Cyan"],
    "large_marker": COLOR_MAP["LightGreen"],
    "large_clamp": COLOR_MAP["LightBlue"],
    "extra_large_clamp": COLOR_MAP["LightCoral"],
    "foam_brick": COLOR_MAP["LightPink"],
}

HO3D_VIDEONAME_TO_OBJ = {
    "AP": "019_pitcher_base",
    "MPM": "010_potted_meat_can",
    "SB": "021_bleach_cleanser",
    "SM": "006_mustard_bottle",
}


def get_obj_info(dataset_name):
    if dataset_name not in ["lm", "ycbv", "tless", "tudl", "hb", "icbin", "itodd"]:
        raise AssertionError("dataset name unknow")
    return {
        "obj_name_to_id": eval("{}_obj_name_to_id".format(dataset_name)),
        "obj_id_to_name": eval("{}_obj_id_to_name".format(dataset_name)),
        "symmetry_obj_names": eval("{}_symmetry_obj_names".format(dataset_name)),
    }
