lm_obj_name_to_id = {
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
lm_obj_id_to_name = {v: k for k, v in lm_obj_name_to_id.items()}
lm_symmetry_obj_names = {"eggbox", "glue", "cup", "bowl"}


ycbv_obj_name_to_id = {
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
ycbv_obj_id_to_name = {v: k for k, v in ycbv_obj_name_to_id.items()}
ycbv_symmetry_obj_names = {"bowl", "wood_block", "large_clamp", "extra_large_clamp", "foam_brick"}

tless_obj_name_to_id = {
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
tl_obj_id_to_name = {v: k for k, v in tless_obj_name_to_id.items()}
tless_symmetry_obj_names = ["obj{:02d}".format(obj_id + 1) for obj_id in range(0, 30)]

tudl_obj_name_to_id = {"obj01": 1, "obj02": 2, "obj03": 3}
tu_obj_id_to_name = {v: k for k, v in tudl_obj_name_to_id.items()}
tudl_symmetry_obj_names = []

hb_obj_name_to_id = {
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
hb_obj_id_to_name = {v: k for k, v in hb_obj_name_to_id.items()}
hb_symmetry_obj_names = ["obj10", "obj11", "obj14"]

icbin_obj_name_to_id = {"obj01": 1, "obj02": 2}
icbin_obj_id_to_name = {v: k for k, v in icbin_obj_name_to_id.items()}
icbin_symmetry_obj_names = ["obj01"]

itodd_obj_name_to_id = {
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
ito_obj_id_to_name = {v: k for k, v in itodd_obj_name_to_id.items()}
itodd_symmetry_obj_names = [
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

ycbineoat_videoname_to_objects = {
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
ycbineoat_symmetry_obj_names = []


def get_obj_info(dataset_name):
    if dataset_name not in ["lm", "ycbv", "tless", "tudl", "hb", "icbin", "itodd"]:
        raise AssertionError("dataset name unknow")
    return {
        "obj_name_to_id": eval("{}_obj_name_to_id".format(dataset_name)),
        "obj_id_to_name": eval("{}_obj_id_to_name".format(dataset_name)),
        "symmetry_obj_names": eval("{}_symmetry_obj_names".format(dataset_name)),
    }
