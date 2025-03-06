
server = 0    # 0: UNINA - 1: MTD - 2: INRIA

if server == 0:
    root = "/home/giuseppe.guarino/Datasets/Datasets_INRIA/"
    results_dir = '/home/giuseppe.guarino/SHeDD/results/'
elif server == 1:
    root = "/home/giuseppe/Datasets/"
    results_dir = '/home/giuseppe/SHeDD/results/'
else:
    root = "/home/gguarino/Datasets"
    results_dir = '/home/gguarino/SHeDD/results/'

ds = ["SUNRGBD", "TRISTAR", "HANDS"]  #datasets

data_names = {
    "SUNRGBD": ["DEPTH", "RGB"],
    "TRISTAR": ["DEPTH", "THERMAL"],
    "HANDS": ["DEPTH", "RGB"],
}

backbones_list = ["ResNet-18", "TinyViT"]

TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-4 #0.03
LEARNING_RATE_DC = 1e-3
MOMENTUM_EMA = .95
EPOCHS =  300
WARM_UP_EPOCH_EMA = 50

EMB_SIZE = 128 #2048
TEMP = 0.01
TEMP_2 = 0.04

GP_PARAM = 10
DC_PARAM = 0.1
ITER_DC = 10 # Inner iterations for domain critic module

decouple_ds = True

TV_param = {
    "embed_dims": [64, 128, 160, 320],
    "depths": [2, 2, 6, 2],
    "num_heads": [2, 4, 5, 10],
    "window_sizes": [7, 7, 14, 7],
    "drop_path_rate": 0.0,
}
