import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
from backbone_resnet import DASH_DA
from backbone_vit import DASH_DA_TV
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.autograd import grad
from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform
import os
from param import ds, data_names, TRAIN_BATCH_SIZE, LEARNING_RATE, LEARNING_RATE_DC, MOMENTUM_EMA, \
    EPOCHS, WARM_UP_EPOCH_EMA, GP_PARAM, DC_PARAM, ITER_DC, TV_param
from tqdm import tqdm


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        if y_batch.shape[0] == TRAIN_BATCH_SIZE:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            _, _ ,_, pred = model.forward_test_target(x_batch)
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


def gradient_penalty(critic, h_s, h_t, device):
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

##########################
# MAIN FUNCTION: TRAINING
##########################
def train_and_eval(ds_path, out_dir, nsamples, nsplit, ds_idx, source_idx, gpu, backbone):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source_prefix = data_names[ds[ds_idx]][source_idx]
    target_prefix = data_names[ds[ds_idx]][source_idx - 1]

    source_data = np.load(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_data_filtered.npy"))
    target_data = np.load(os.path.join(ds_path, ds[ds_idx], f"{target_prefix}_data_filtered.npy"))
    source_label = np.load(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_label_filtered.npy"))
    target_label = np.load(os.path.join(ds_path, ds[ds_idx], f"{target_prefix}_label_filtered.npy"))

    sys.stdout.flush()
    train_target_idx = np.load( os.path.join(ds_path, ds[ds_idx], "train_idx", f"{target_prefix}_{nsplit}_{nsamples}_train_idx.npy") )
    test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

    train_target_data = target_data[train_target_idx]
    train_target_label = target_label[train_target_idx]

    test_target_data = target_data[test_target_idx]
    test_target_label = target_label[test_target_idx]

    test_target_data_unl = target_data[test_target_idx]

    n_classes = len(np.unique(source_label))

    TR_BATCH_SIZE = np.minimum(int(n_classes * nsamples), TRAIN_BATCH_SIZE)
    TR_BATCH_SIZE = int(TR_BATCH_SIZE)

    source_data, source_label = shuffle(source_data, source_label)
    train_target_data, train_target_label = shuffle(train_target_data, train_target_label)

    #DATALOADER SOURCE
    x_train_source = torch.tensor(source_data, dtype=torch.float32)
    y_train_source = torch.tensor(source_label, dtype=torch.int64)

    dataset_source = MyDataset(x_train_source, y_train_source, transform=transform)
    dataloader_source = DataLoader(dataset_source, shuffle=True, batch_size=TR_BATCH_SIZE)

    #DATALOADER TARGET TRAIN
    x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
    y_train_target = torch.tensor(train_target_label, dtype=torch.int64)

    dataset_train_target = MyDataset(x_train_target, y_train_target, transform=transform)
    dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=TR_BATCH_SIZE//2)

    #DATALOADER TARGET UNLABELLED
    x_train_target_unl = torch.tensor(test_target_data_unl, dtype=torch.float32)

    dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform)
    dataloader_train_target_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=TR_BATCH_SIZE//2)

    #DATALOADER TARGET TEST
    x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
    y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
    dataset_test_target = TensorDataset(x_test_target, y_test_target)
    dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

    if backbone == "ResNet-18":
        model = DASH_DA(input_channel_source=source_data.shape[1],
                        input_channel_target=target_data.shape[1],
                        num_classes=n_classes)
        model = model.to(device)

        critic = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).to(device)

    elif backbone == "TinyViT":
        model = DASH_DA_TV(img_size_source=source_data.shape[2], in_chans_source=source_data.shape[1],
                         img_size_target=target_data.shape[2], in_chans_target=target_data.shape[1],
                         num_classes=n_classes,
                         embed_dims=TV_param["embed_dims"], depths=TV_param["depths"],
                         num_heads=TV_param["num_heads"], window_sizes=TV_param["window_sizes"],
                         drop_path_rate=TV_param["drop_path_rate"]
                         )
        model = model.to(device)

        critic = nn.Sequential(
            nn.Linear(160, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).to(device)

    else:
        print("Backbone not found!")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=LEARNING_RATE_DC)

    pbar = tqdm(range(EPOCHS))

    ema_weights = None
    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, EPOCHS))
        start = time.time()
        model.train()
        tot_loss = 0.0
        tot_ortho_loss = 0.0
        den = 0
        for x_batch_source, y_batch_source in dataloader_source:

            if x_batch_source.shape[0] < TR_BATCH_SIZE:
                continue  # To avoid errors on pairing source/target samples
            optimizer.zero_grad()
            x_batch_target, y_batch_target = next(iter(dataloader_train_target))
            x_batch_target_unl, x_batch_target_unl_aug = next(iter(dataloader_train_target_unl))

            x_batch_source = x_batch_source.to(device)
            y_batch_source = y_batch_source.to(device)

            x_batch_target = x_batch_target.to(device)
            y_batch_target = y_batch_target.to(device)

            x_batch_target_unl = x_batch_target_unl.to(device)
            x_batch_target_unl_aug = x_batch_target_unl_aug.to(device)

            # TRAIN DOMAIN CRITIC
            set_requires_grad(model, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)
            with torch.no_grad():
                h_s, _, _, _ = model.forward_source(x_batch_source, 0)
                h_t, _, _, _ = model.forward_source(torch.cat((x_batch_target, x_batch_target_unl_aug), 0), 1)

            for _ in range(ITER_DC):
                gp = gradient_penalty(critic, h_s, h_t, device)

                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + GP_PARAM * gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

            # TRAIN CLASSIFIER
            set_requires_grad(model, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)

            emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl, emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl = model(
                [x_batch_source, x_batch_target])

            pred_task = torch.cat([task_source_cl, task_target_cl],dim=0)
            pred_dom = torch.cat([dom_source_cl, dom_target_cl],dim=0)
            y_batch = torch.cat([y_batch_source, y_batch_target],dim=0)
            y_batch_dom = torch.cat([torch.zeros_like(y_batch_source), torch.ones_like(y_batch_target)],dim=0)

            loss_pred = loss_fn(pred_task, y_batch)
            loss_dom = loss_fn( pred_dom, y_batch_dom)

            inv_emb = torch.cat([emb_source_inv, emb_target_inv])
            spec_emb = torch.cat([emb_source_spec, emb_target_spec])

            norm_inv_emb = nn.functional.normalize(inv_emb)
            norm_spec_emb = nn.functional.normalize(spec_emb)
            loss_ortho = torch.sum( norm_inv_emb * norm_spec_emb, dim=1)
            loss_ortho = torch.mean(loss_ortho)

            model.target.train()
            unl_target_inv, unl_target_spec, pred_unl_target_dom, pred_unl_target = model.forward_source(x_batch_target_unl, 1)
            unl_target_aug_inv, unl_target_aug_spec, pred_unl_target_strong_dom, pred_unl_target_strong = model.forward_source(x_batch_target_unl_aug, 1)

            pred_unl_dom = torch.cat([pred_unl_target_strong_dom,pred_unl_target_dom],dim=0)
            u_loss_dom = loss_fn(pred_unl_dom, torch.ones(pred_unl_dom.shape[0]).long().to(device))

            unl_inv = torch.cat([unl_target_inv,unl_target_aug_inv],dim=0)
            norm_unl_inv = F.normalize(unl_inv)
            unl_spec = torch.cat([unl_target_spec,unl_target_aug_spec],dim=0)
            norm_unl_spec = F.normalize(unl_spec)
            u_loss_ortho = torch.mean( torch.sum( norm_unl_inv * norm_unl_spec, dim=1) )

            emb_t_all = torch.cat((emb_target_inv, unl_target_aug_inv), dim=0)  # all target embeddings (labelled + unlabelled)
            wasserstein_distance = critic(emb_source_inv).mean() - critic(emb_t_all).mean()

            loss = loss_pred + loss_dom + loss_ortho + u_loss_dom + u_loss_ortho + DC_PARAM * wasserstein_distance

            loss.backward()
            optimizer.step()

            tot_loss+= loss.cpu().detach().numpy()
            tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
            den+=1.

        end = time.time()
        pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
        f1_val = f1_score(labels_valid, pred_valid, average="weighted")
        
        ####################### EMA #####################################
        f1_val_ema = 0
        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
            current_state_dict = model.state_dict()
            model.load_state_dict(ema_weights)
            pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
            f1_val_ema = f1_score(labels_valid, pred_valid, average="weighted")
            model.load_state_dict(current_state_dict)
        ####################### EMA #####################################

        pbar.set_postfix(
            {'Loss': tot_loss/den, 'F1 (ORIG)': 100*f1_val, 'F1 (EMA)': 100*f1_val_ema, 'Time': (end-start)})
        sys.stdout.flush()

    #Create folder to save model weights
    model_dir = os.path.join(out_dir, ds[ds_idx], "models", source_prefix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    output_file = os.path.join( model_dir, f"{source_prefix}_{nsplit}_{nsamples}.pth" )
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)

    return 100*f1_val_ema