import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from datasets import AQADataset
from models.coqm import preserve_torch_rng_state
from utils.utils import AverageMeter


EVAL_POLICIES = ("legacy_gumbel", "argmax", "mc_gumbel")


def build_dataset(
        dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat,
        batch_size=32, num_workers=2, segments=-1):
    """Build the official full train/test loaders without split overrides."""
    train_dataset = AQADataset(
        dataset_name=dataset_name,
        feat_dir=feat_dir,
        rgb_feat=rgb_feat,
        flow_feat=flow_feat,
        audio_feat=audio_feat,
        squeeze_rgb_feat="mean",
        segments=segments,
        is_train=True,
        sequence_sampling="legacy",
    )
    test_dataset = AQADataset(
        dataset_name=dataset_name,
        feat_dir=feat_dir,
        rgb_feat=rgb_feat,
        flow_feat=flow_feat,
        audio_feat=audio_feat,
        squeeze_rgb_feat="mean",
        segments=segments,
        is_train=False,
        sequence_sampling="legacy",
    )
    # Intentionally match the official repository: no private generator and
    # no custom worker_init_fn. This preserves its shuffle/crop RNG path.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def build_model(model_name, model_conf=None):
    if model_name == "pamfn_base":
        from models.pamfn import BaseModel
        return BaseModel(**model_conf)
    if model_name == "pamfn_final":
        from models.pamfn import PAMFN
        return PAMFN(**model_conf)
    raise ValueError(f"model: {model_name} is not supported")


def build_optimizer(
        model, optim, lr, momentum=0.9, weight_decay=None,
        reduce_fc_lr=False, model_name=None):
    if reduce_fc_lr and model_name == "pamfn_final":
        fc_ids = {id(parameter) for parameter in model.fc.parameters()}
        other_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in fc_ids
        ]
        parameter_groups = [
            {"params": model.fc.parameters(), "lr": lr * 0.1},
            {"params": other_parameters},
        ]
    else:
        parameter_groups = model.parameters()

    if optim == "Adam":
        return torch.optim.Adam(parameter_groups, lr=lr)
    if optim == "SGD":
        return torch.optim.SGD(
            parameter_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optim == "AdamW":
        # The official code does not pass its configured weight_decay here.
        return torch.optim.AdamW(parameter_groups, lr=lr)
    raise ValueError(f"optimizer: {optim} is not supported")


def build_lr_scheduler(
        optimizer, scheduler, epoch_num, scheduler_steps,
        lr_min=5e-6, warmup_lr_init=5e-6, decay_rate=0.1):
    from utils.lr_scheduler import build_scheduler

    return build_scheduler(
        optimizer,
        scheduler,
        epoch_num,
        scheduler_steps,
        decay_rate=decay_rate,
        lr_min=lr_min,
        warmup_lr_init=warmup_lr_init,
    )


def _device_for(model):
    return next(model.parameters()).device


def _prepare_batch(batch, device):
    video, audio, flow, label, _ = batch
    return {
        "V": video.to(device),
        "F": flow.to(device),
        "A": audio.to(device),
    }, label.float().to(device)


def _spearman(predictions, labels):
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    coefficient, _ = spearmanr(prediction, target)
    return float(coefficient)


def train_epoch(
        model, optimizer, lr_scheduler, dataloader,
        current_epoch, clip_grad=False):
    objective_losses = AverageMeter()
    mse_losses = AverageMeter()
    predictions = []
    labels = []
    start = time.time()
    device = _device_for(model)
    model.train()

    for batch in dataloader:
        model_input, label = _prepare_batch(batch, device)
        prediction, other_info = model(model_input)
        training_loss = getattr(
            model, "call_training_loss", model.call_loss
        )
        objective = training_loss(
            prediction, label, **other_info
        )
        mse = torch.mean(
            (prediction.reshape(-1) - label.reshape(-1)) ** 2
        )

        optimizer.zero_grad()
        objective.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        batch_size = label.shape[0]
        objective_losses.update(objective.item(), batch_size)
        mse_losses.update(mse.detach().item(), batch_size)
        predictions.extend(
            prediction.detach().cpu().reshape(-1).tolist()
        )
        labels.extend(label.detach().cpu().reshape(-1).tolist())

    if lr_scheduler is not None:
        lr_scheduler.step(current_epoch + 1)
    return {
        "time": time.time() - start,
        "mse_x100": mse_losses.avg * 100,
        "objective_x100": objective_losses.avg * 100,
        # Match the official training-side scale in the epoch log.
        "coef": _spearman(predictions, labels),
    }


def _validate_eval_config(eval_policy, eval_mc_samples):
    if eval_policy not in EVAL_POLICIES:
        raise ValueError(
            "eval_policy must be legacy_gumbel, argmax, or mc_gumbel"
        )
    if eval_mc_samples < 1:
        raise ValueError("eval_mc_samples must be positive")
    if eval_policy == "mc_gumbel" and eval_mc_samples < 2:
        raise ValueError(
            "mc_gumbel requires at least two evaluation samples"
        )


def _seed_torch_for_evaluation(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_epoch(
        model, dataloader, eval_policy="legacy_gumbel",
        eval_mc_samples=20, eval_seed=20260908):
    _validate_eval_config(eval_policy, eval_mc_samples)
    model_eval_policy = getattr(model, "eval_policy", eval_policy)
    if model_eval_policy != eval_policy:
        raise ValueError(
            "model eval_policy does not match the evaluation request: "
            f"{model_eval_policy} != {eval_policy}"
        )
    eval_passes = (
        eval_mc_samples if eval_policy == "mc_gumbel" else 1
    )
    losses = AverageMeter()
    predictions = []
    labels = []
    coqm_curve_cosines = []
    coqm_probabilities = []
    coqm_coordinates = []
    start = time.time()
    device = _device_for(model)
    model.eval()

    rng_context = (
        preserve_torch_rng_state()
        if eval_policy != "legacy_gumbel" else nullcontext()
    )
    with rng_context:
        if eval_policy != "legacy_gumbel":
            _seed_torch_for_evaluation(eval_seed)
        with torch.no_grad():
            for batch in dataloader:
                model_input, label = _prepare_batch(batch, device)
                pass_predictions = []
                pass_coqm_outputs = []
                for _ in range(eval_passes):
                    prediction, other_info = model(model_input)
                    pass_predictions.append(prediction)
                    coqm_output = other_info.get("coqm_output")
                    if coqm_output is not None:
                        pass_coqm_outputs.append(coqm_output)

                prediction = torch.stack(
                    pass_predictions, dim=0
                ).mean(dim=0)
                loss = model.call_loss(prediction, label)
                losses.update(loss.item(), label.shape[0])
                predictions.extend(
                    prediction.detach().cpu().reshape(-1).tolist()
                )
                labels.extend(label.detach().cpu().reshape(-1).tolist())

                if pass_coqm_outputs:
                    if len(pass_coqm_outputs) != eval_passes:
                        raise RuntimeError(
                            "COQM output was missing from an evaluation pass"
                        )
                    curve_cosines = []
                    for coqm_output in pass_coqm_outputs:
                        embedding = coqm_output["embedding"]
                        curve_target = model.coqm.manifold_point(
                            label.reshape(-1).to(
                                dtype=embedding.dtype
                            )
                        )
                        curve_cosines.append(torch.sum(
                            embedding * curve_target, dim=1
                        ))
                    coqm_curve_cosines.extend(
                        torch.stack(curve_cosines, dim=0)
                        .mean(dim=0).detach().cpu().tolist()
                    )
                    coqm_probabilities.extend(
                        torch.stack([
                            output["probabilities"]
                            for output in pass_coqm_outputs
                        ], dim=0).mean(dim=0).detach().cpu().tolist()
                    )
                    coqm_coordinates.extend(
                        torch.stack([
                            output["expected_quality"]
                            for output in pass_coqm_outputs
                        ], dim=0).mean(dim=0).detach().cpu().tolist()
                    )

    prediction_array = np.asarray(predictions, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    target_std = float(np.std(label_array))
    result = {
        "time": time.time() - start,
        "evaluation": {
            "policy": eval_policy,
            "passes": eval_passes,
            "seed": (
                eval_seed if eval_policy != "legacy_gumbel" else None
            ),
            "rng_isolated": eval_policy != "legacy_gumbel",
        },
        "loss_x100": losses.avg * 100,
        "coef": _spearman(predictions, labels) * 100,
        "prediction_std": float(np.std(prediction_array)),
        "target_std": target_std,
        "prediction_std_ratio": (
            float(np.std(prediction_array) / target_std)
            if target_std > 0 else float("nan")
        ),
        "mae": float(np.mean(np.abs(prediction_array - label_array))),
        "bias": float(np.mean(prediction_array - label_array)),
    }
    if coqm_coordinates:
        probabilities = np.asarray(
            coqm_probabilities, dtype=np.float64
        )
        coordinates = np.asarray(coqm_coordinates, dtype=np.float64)
        hard_counts = np.bincount(
            probabilities.argmax(axis=1), minlength=probabilities.shape[1]
        ).astype(np.float64)
        prototypes = model.coqm.manifold_point(
            model.coqm.anchor_scores
        ).detach()
        u, v = model.coqm.orthogonal_basis()
        result["coqm"] = {
            "curve_cosine": float(np.mean(coqm_curve_cosines)),
            "coordinate_mae": float(
                np.mean(np.abs(coordinates - label_array))
            ),
            "quality_std_ratio": (
                float(np.std(coordinates) / target_std)
                if target_std > 0 else float("nan")
            ),
            "soft_occupancy": probabilities.mean(axis=0).tolist(),
            "hard_occupancy": (
                hard_counts / hard_counts.sum()
            ).tolist(),
            "orthogonal_error": float(
                torch.abs(torch.sum(u * v)).cpu().item()
            ),
            "adjacent_cosine_distance": (
                1.0 - torch.sum(
                    prototypes[:-1] * prototypes[1:], dim=1
                )
            ).cpu().tolist(),
            "score_q_spearman": _spearman(
                prediction_array, coordinates
            ),
            "score_q_pearson": float(np.corrcoef(
                prediction_array, coordinates
            )[0, 1]),
        }
    return result


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def train_loop(
        model, optimizer, lr_scheduler,
        train_dataloader, test_dataloader, epoch_num,
        save_base_model="", save_final_model="", save_moniter="",
        log_func=None, clip_grad=False,
        eval_policy="legacy_gumbel", eval_mc_samples=20,
        eval_seed=20260908):
    log_func = log_func or print
    max_coef = 0.0
    max_avg_coef = 0.0
    min_loss = 10.0
    min_avg_loss = 10.0
    coefficient_of_saved_model = 0.0
    test_losses = []
    test_coefs = []

    for current_epoch in range(epoch_num):
        train_result = train_epoch(
            model,
            optimizer,
            lr_scheduler,
            train_dataloader,
            current_epoch,
            clip_grad=clip_grad,
        )
        test_result = test_epoch(
            model,
            test_dataloader,
            eval_policy=eval_policy,
            eval_mc_samples=eval_mc_samples,
            eval_seed=eval_seed,
        )
        test_losses.append(test_result["loss_x100"])
        test_coefs.append(test_result["coef"])
        average_loss = np.mean(test_losses[-10:])
        average_coef = np.mean(test_coefs[-10:])

        if save_base_model:
            better = (
                save_moniter == "avg_coef"
                and average_coef > max_avg_coef
            ) or (
                save_moniter == "avg_loss"
                and average_loss < min_avg_loss
            )
            if better:
                _ensure_parent(save_base_model)
                torch.save(model, save_base_model)
                coefficient_of_saved_model = test_result["coef"]

        if save_final_model and (
                current_epoch == 0 or test_result["coef"] > max_coef):
            _ensure_parent(save_final_model)
            torch.save(model.state_dict(), save_final_model)
            coefficient_of_saved_model = test_result["coef"]

        max_coef = max(max_coef, test_result["coef"])
        max_avg_coef = max(max_avg_coef, average_coef)
        min_loss = min(min_loss, test_result["loss_x100"])
        min_avg_loss = min(min_avg_loss, average_loss)
        log_func(
            "Epoch[{0}/{1}] \t"
            "Time: {train_time:.1f}/{test_time:.1f} \t"
            "Loss {train_loss:.4f}/{test_loss:.4f} \t"
            "Objective {objective:.4f} \t"
            "Avg_loss {avg_loss:.2f}/{min_avg_loss:.2f}\t"
            "Coef {train_coef:.2f}/{test_coef:.2f} \t"
            "Avg_coef {avg_coef:.2f}/{max_avg_coef:.2f} \t"
            "BestCoef {max_coef:.2f}\t BestLoss {min_loss:.2f}\t".format(
                current_epoch,
                epoch_num,
                train_time=train_result["time"],
                test_time=test_result["time"],
                train_loss=train_result["mse_x100"],
                test_loss=test_result["loss_x100"],
                objective=train_result["objective_x100"],
                avg_loss=average_loss,
                min_avg_loss=min_avg_loss,
                train_coef=train_result["coef"],
                test_coef=test_result["coef"],
                avg_coef=average_coef,
                max_avg_coef=max_avg_coef,
                max_coef=max_coef,
                min_loss=min_loss,
            )
        )

    return (
        max_avg_coef,
        max_coef,
        min_avg_loss,
        min_loss,
        coefficient_of_saved_model,
    )


def go(
        model_kwargs, optimizer_kwargs, lr_scheduler_kwargs,
        dataset_kwargs, epoch_num, save_base_model="",
        save_final_model="", save_moniter="", log_func=None,
        seed=0, clip_grad=False, eval_policy="legacy_gumbel",
        eval_mc_samples=20, eval_seed=20260908):
    log_func = log_func or print
    setup_seed(seed)
    train_loader, test_loader = build_dataset(**dataset_kwargs)
    model = build_model(**model_kwargs)
    optimizer = build_optimizer(
        model,
        model_name=model_kwargs["model_name"],
        **optimizer_kwargs,
    )
    scheduler = build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    results = train_loop(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        epoch_num=epoch_num,
        save_base_model=save_base_model,
        save_moniter=save_moniter,
        save_final_model=save_final_model,
        log_func=log_func,
        clip_grad=clip_grad,
        eval_policy=eval_policy,
        eval_mc_samples=eval_mc_samples,
        eval_seed=eval_seed,
    )
    log_func(
        f"best avg coef: {results[0] / 100:.4f}\t"
        f"best coef: {results[1] / 100:.4f}\t"
        f"min avg loss: {results[2] / 100:.6f}\t"
        f"min loss: {results[3] / 100:.6f}\t"
        "coef of saved model: "
        f"{results[4] / 100:.4f}"
    )
    return results


def _load_state_dict(path):
    try:
        value = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value:
        return value["state_dict"]
    return value


def test_model(
        model_kwargs, dataset_kwargs, ckpt_path="",
        log_func=None, seed=0, eval_policy="legacy_gumbel",
        eval_mc_samples=20, eval_seed=20260908):
    log_func = log_func or print
    setup_seed(seed)
    _, test_loader = build_dataset(**dataset_kwargs)
    model = build_model(**model_kwargs)
    model.load_state_dict(_load_state_dict(ckpt_path))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    result = test_epoch(
        model,
        test_loader,
        eval_policy=eval_policy,
        eval_mc_samples=eval_mc_samples,
        eval_seed=eval_seed,
    )
    log_func(
        f"Coef: {result['coef'] / 100:.4f}\t"
        f"Loss: {result['loss_x100'] / 100:.6f}\t"
        f"PredStd/GTStd: {result['prediction_std_ratio']:.4f}"
    )
    if "coqm" in result:
        coqm = result["coqm"]
        log_func(
            f"COQM curve cosine: {coqm['curve_cosine']:.4f}\t"
            f"coord MAE: {coqm['coordinate_mae']:.6f}\t"
            f"q std/GT std: {coqm['quality_std_ratio']:.4f}\t"
            f"score/q SRCC: {coqm['score_q_spearman']:.4f}"
        )
    return result


def print_flops(model_kwargs):
    from thop import clever_format, profile

    model = build_model(**model_kwargs)
    data_in = {
        "V": torch.zeros(1, 70, 1024),
        "F": torch.zeros(1, 70, 1024),
        "A": torch.zeros(1, 70, 768),
    }
    macs, params = profile(model, inputs=(data_in,))
    return clever_format([macs, params], "%.3f")


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_info(info, logger=None, print_log=True):
    if logger is not None:
        logger.info(info)
    elif print_log:
        print(info)
