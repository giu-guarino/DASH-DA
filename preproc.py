import numpy as np
from sklearn.utils import shuffle
import os
from param import ds, data_names, decouple_ds


def getIdxVal(sub_hashCl2idx, val):
    idx = []
    for k in sub_hashCl2idx.keys():
        temp = sub_hashCl2idx[k]
        idx.append(temp[0:val])
    return np.concatenate(idx, axis=0)


def get_idxPerClass(hashCl2idx, max_val):

    sub_hashCl2idx = {}
    for k in hashCl2idx.keys():
        temp = hashCl2idx[k]
        temp = shuffle(temp)
        sub_hashCl2idx[k] = temp[0:max_val]
    return sub_hashCl2idx


def extractWriteTrainIdx(root, nrepeat, nsample_list, hashCl2idx, dataset, modality):
    max_val = nsample_list[-1]
    for i in range(nrepeat):
        sub_hashCl2idx = get_idxPerClass(hashCl2idx, max_val)
        for val in nsample_list:
            idx = getIdxVal(sub_hashCl2idx, val)
            np.save(os.path.join(root, dataset, "train_idx", "%s_%d_%d_train_idx.npy"%(modality,i,val)), idx)

def getHash2classes(labels):
    hashCl2idx = {}
    for v in np.unique(labels):
        idx = np.where(labels == v)[0]
        idx = shuffle(idx)
        hashCl2idx[v] = idx
    return hashCl2idx
        
def writeFilteredData(root, dataset, modality, data, label):
    np.save(os.path.join(root, dataset, "%s_data_filtered.npy"%modality), data)
    np.save(os.path.join(root, dataset, "%s_label_filtered.npy"%modality), label)


for ds_idx in [0, 1, 2]: # 0: SUNRGBD, 1: TRISTAR, 2: HANDS

    ds_dir = "Datasets"

    print(f"Processing {ds[ds_idx]}")

    out_dir = os.path.join(ds_dir, ds[ds_idx], "train_idx")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #READ DATA
    data_mod1 = np.load(os.path.join(ds_dir, ds[ds_idx], f"{data_names[ds[ds_idx]][0]}_data_normalized.npy")).astype("float32")
    data_mod2 = np.load(os.path.join(ds_dir, ds[ds_idx], f"{data_names[ds[ds_idx]][1]}_data_normalized.npy")).astype("float32")

    if decouple_ds:

        # SELECT UNCORRELATED DATA
        labels = np.load(os.path.join(ds_dir, ds[ds_idx], f"labels.npy"))

        mod1_idx = np.arange(0, labels.shape[0], 2)
        mod2_idx = np.arange(1, labels.shape[0], 2)

        data_mod1 = data_mod1[mod1_idx]
        label_mod1 = labels[mod1_idx]

        data_mod2 = data_mod2[mod2_idx]
        label_mod2 = labels[mod2_idx]

    else:
        labels = np.load(os.path.join(ds_dir, ds[ds_idx], f"labels.npy"))

        label_mod1 = labels
        label_mod2 = labels


    #WRITE DATA
    writeFilteredData(ds_dir, ds[ds_idx], data_names[ds[ds_idx]][0], data_mod1, label_mod1)
    writeFilteredData(ds_dir, ds[ds_idx], data_names[ds[ds_idx]][1], data_mod2, label_mod2)

    mod1_hashCl2idx = getHash2classes(label_mod1)
    mod2_hashCl2idx = getHash2classes(label_mod2)

    #EXTRACT 5 time TRAIN IDX INCREASING THE NUMBER OF SAMLPE PER CLASS FROM 5 TO 50
    nrepeat = 5
    nsample_list = [5, 10, 25, 50]
    extractWriteTrainIdx(ds_dir, nrepeat, nsample_list, mod1_hashCl2idx, ds[ds_idx], data_names[ds[ds_idx]][0])
    extractWriteTrainIdx(ds_dir, nrepeat, nsample_list, mod2_hashCl2idx, ds[ds_idx], data_names[ds[ds_idx]][1])



