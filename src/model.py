import os

import torch
import torch.nn as nn
import torchvision.models as models

from models.simclr_resnet import get_simclr_resnet


def get_model(args) -> nn.Module:
    """
    Wrapper to get the model based on args.arch.
    Currently, this function only supports 'r50_1x_sk0'.
    For future expansion, add logic to handle different args.arch values.
    """
    model_name = args.arch
    checkpoint_path = args.pretrain
    num_classes = args.num_classes
    num_channels = args.num_channels

    if model_name == "r50_1x_sk0":
        resnet, contrastive_head = get_simclr_resnet(
            depth=50, width_multiplier=1, sk_ratio=0
        )
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
        if num_channels != 3:
            resnet.net[0][0].in_channels = num_channels
            resnet.net[0][0].weight = torch.nn.Parameter(
                resnet.net[0][0].weight[:, :num_channels, :, :]
            )
        if not hasattr(args, "pretrain"):
            raise ValueError(
                "Pretrained model path not provided. Please provide the path to the pretrained model."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained model not found at: {checkpoint_path}")
        load_state_dict_subset(resnet, torch.load(checkpoint_path, weights_only=True))
        return resnet
    else:
        raise NotImplementedError(
            f"Model architecture '{model_name}' is not yet implemented in this demo."
        )


def load_state_dict_subset(model, state: dict, verbose=True):
    # only keep weights whose shapes match the corresponding layers in the model
    if "state_dict" in state:
        state = state["state_dict"]
    elif "resnet" in state:  # simclrv2 weights split into "resnet" and "head"
        state = state["resnet"]
    elif "head" in state:
        state = state["head"]
    elif "model" in state:
        state = state["model"]
    m_s = model.state_dict()
    if not verbose:
        subset = {
            k: v for k, v in state.items() if k in m_s and m_s[k].shape == v.shape
        }
    else:
        subset = dict()
        for k, v in state.items():
            if k in m_s and m_s[k].shape == v.shape:
                subset[k] = v
            elif k not in m_s:
                print(f"{k} not in model")
            elif k in m_s and m_s[k].shape != v.shape:
                print(f"{k} in model but shape {m_s[k].shape} != {v.shape}")
            else:
                print(f"{k} in model, but couldn't load for some reason.")
    print(
        f"Loaded {len(subset)}/{len(state)} weights for {model.__class__.__name__} (which has {len(m_s)} parameters)."
    )
    model.load_state_dict(subset, strict=False)
    return subset
