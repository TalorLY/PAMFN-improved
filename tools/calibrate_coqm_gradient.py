#!/usr/bin/env python
"""Measure COQM-to-score gradient scale on one fixed training batch."""

import argparse
import json
import math
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--action", default="TES")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feat_dir", default="./data/features")
    parser.add_argument("--weight", type=float, default=1e-4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_config(action):
    if action in ("Ball", "Clubs", "Hoop", "Ribbon"):
        import configs.RG_feats2 as config
    else:
        import configs.FISV_feats2 as config
    return config


def main():
    args = parse_args()
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    import torch

    from pipeline import (
        _prepare_batch,
        build_dataset,
        build_model,
        setup_seed,
    )

    if not math.isfinite(args.weight) or args.weight <= 0:
        raise ValueError("weight must be finite and positive")
    config = load_config(args.action)
    current = config.multi_modality
    dataset_config = {
        "dataset_name": args.action,
        "feat_dir": args.feat_dir,
        "rgb_feat": config.features["V"],
        "flow_feat": config.features["F"],
        "audio_feat": config.features["A"],
        "batch_size": current["batch_size"],
        "num_workers": current["num_workers"],
    }
    model_config = {
        "model_name": "pamfn_final",
        "model_conf": {
            "model_dim": current["model_dim"],
            "fc_drop": current["fc_drop"],
            "fc_r": current["fc_r"],
            "feat_drop": current["feat_drop"],
            "K": current["K"],
            "ms_heads": current["ms_heads"],
            "cm_heads": current["cm_heads"],
            "ckpt_dir": current["ckpt_dir"],
            "rgb_ckpt_name": current["rgb_ckpt_name"],
            "flow_ckpt_name": current["flow_ckpt_name"],
            "audio_ckpt_name": current["audio_ckpt_name"],
            "dataset_name": args.action,
            "coqm_weight": args.weight,
            "coqm_dim": 128,
            "coqm_temperature": 0.1,
            "coqm_theta_max": math.pi / 3,
            "coqm_coord_scale": 1.0,
            "quality_aux_rng_isolated": True,
        },
    }

    setup_seed(args.seed)
    train_loader, _ = build_dataset(**dataset_config)
    model = build_model(**model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    features = {}

    def capture_feature(_, values):
        features["stage3"] = values[0]

    # The score head and COQM branch both consume this exact stage-3 tensor.
    # A hook on the projector would capture only a downstream squeeze view,
    # which is not an ancestor of score_loss in the autograd graph.
    hook = model.fc.register_forward_pre_hook(capture_feature)
    batch = next(iter(train_loader))
    model_input, label = _prepare_batch(batch, device)
    prediction, info = model(model_input)
    hook.remove()
    stage3 = features["stage3"]
    score_loss = model.mse(prediction.reshape(-1), label.reshape(-1))
    components = model.coqm.loss_components(
        info["coqm_output"], label, coord_scale=1.0
    )
    score_gradient = torch.autograd.grad(
        score_loss, stage3, retain_graph=True
    )[0]
    aux_gradient = torch.autograd.grad(
        components["total"], stage3
    )[0]
    score_norm = score_gradient.reshape(score_gradient.shape[0], -1).norm(
        p=2, dim=1
    )
    aux_norm = aux_gradient.reshape(aux_gradient.shape[0], -1).norm(
        p=2, dim=1
    )
    valid = score_norm > 1e-12
    if not bool(valid.any()):
        raise RuntimeError("score gradient is zero for the calibration batch")
    unweighted_ratios = (aux_norm[valid] / score_norm[valid]).detach().cpu()
    weighted_ratios = unweighted_ratios * args.weight
    median_unweighted = float(torch.median(unweighted_ratios).item())
    median_weighted = float(torch.median(weighted_ratios).item())
    result = {
        "action": args.action,
        "seed": args.seed,
        "batch_size": int(label.shape[0]),
        "weight": args.weight,
        "median_unweighted_aux_to_score_gradient": median_unweighted,
        "median_weighted_aux_to_score_gradient": median_weighted,
        "recommended_weight_for_ratio_0p10": (
            0.10 / median_unweighted
            if median_unweighted > 0 else None
        ),
        "target_ratio": 0.10,
        "acceptance_interval": [0.05, 0.20],
        "within_acceptance_interval": (
            0.05 <= median_weighted <= 0.20
        ),
        "score_loss": float(score_loss.detach().cpu().item()),
        "coqm_total": float(components["total"].detach().cpu().item()),
        "coqm_curve": float(components["curve"].detach().cpu().item()),
        "coqm_anchor": float(components["anchor"].detach().cpu().item()),
        "coqm_coord": float(components["coord"].detach().cpu().item()),
        "finite": bool(np.isfinite(median_weighted)),
    }
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
