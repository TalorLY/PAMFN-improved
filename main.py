import argparse
import json
import math
import os
from functools import partial

from datasets import (
    get_train_quality_centers,
    get_train_quality_thresholds,
)
from pipeline import go, log_info, print_flops, test_model


ACTIONS = ("TES", "PCS", "Ball", "Clubs", "Hoop", "Ribbon")


def parser_args():
    parser = argparse.ArgumentParser(
        description="Official PAMFN Base with optional ranking supervision"
    )
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--feats", type=int, required=True, choices=(1, 2))
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--multi_modality", action="store_true")
    parser.add_argument("--modality", default="V", choices=("V", "F", "A"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--feat_dir", default="./data/features")
    parser.add_argument(
        "--sequence_sampling",
        choices=("legacy",),
        default="legacy",
        help="official random-crop/full-test sequence behavior",
    )
    parser.add_argument(
        "--eval_policy",
        choices=("legacy_gumbel", "argmax", "mc_gumbel"),
        default="argmax",
        help=(
            "fusion routing used during evaluation; training remains "
            "stochastic Gumbel"
        ),
    )
    parser.add_argument(
        "--eval_mc_samples",
        type=int,
        default=20,
        help="number of prediction draws averaged by mc_gumbel",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=20260908,
        help="fixed isolated RNG seed for non-legacy evaluation",
    )
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--save_base_model", default="")
    parser.add_argument("--save_final_model", default="")
    parser.add_argument(
        "--result_json", default="",
        help="optional structured training or evaluation result path",
    )
    parser.add_argument(
        "--run_tag", default="",
        help="optional experiment method tag stored in result JSON",
    )
    parser.add_argument(
        "--shared_rank_weight",
        type=float,
        default=0.0,
        help=(
            "training-only pairwise ranking weight; zero exactly recovers "
            "the official PAMFN MSE objective"
        ),
    )
    parser.add_argument(
        "--shared_rank_margin",
        type=float,
        default=0.02,
        help="ignore pairs at or below this normalized label gap",
    )
    parser.add_argument(
        "--shared_rank_temperature",
        type=float,
        default=0.05,
        help="temperature of the logistic pairwise ranking loss",
    )
    parser.add_argument(
        "--shared_list_weight",
        type=float,
        default=0.0,
        help=(
            "training-only grouped ListMLE weight; zero exactly recovers "
            "the official PAMFN MSE objective"
        ),
    )
    parser.add_argument(
        "--shared_list_margin",
        type=float,
        default=0.02,
        help="merge near-tied labels within this normalized score range",
    )
    parser.add_argument(
        "--shared_list_temperature",
        type=float,
        default=0.05,
        help="temperature of the grouped ListMLE loss",
    )
    parser.add_argument(
        "--ordinal_prototype_weight",
        type=float,
        default=0.0,
        help=(
            "training-only ordered quality prototype weight; zero "
            "preserves the official PAMFN objective and state structure"
        ),
    )
    parser.add_argument(
        "--ordinal_prototype_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--ordinal_prototype_temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ordinal_prototype_order_margin",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ordinal_prototype_order_scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ordinal_prototype_target_mode",
        choices=("discrete", "continuous"),
        default="discrete",
    )
    parser.add_argument(
        "--quality_aux_rng_isolated",
        action="store_true",
        help="restore global RNG after auxiliary-head initialization",
    )
    parser.add_argument(
        "--coqm_weight", type=float, default=0.0,
        help="training-only continuous ordered manifold weight",
    )
    parser.add_argument("--coqm_dim", type=int, default=128)
    parser.add_argument(
        "--coqm_temperature", type=float, default=0.1
    )
    parser.add_argument(
        "--coqm_theta_max", type=float, default=math.pi / 3
    )
    parser.add_argument(
        "--coqm_coord_scale", type=float, default=1.0,
        help="set to zero for the pre-registered COQM-NoCoord ablation",
    )
    parser.add_argument(
        "--coqm_anchor_mode",
        choices=("uniform", "train_centers"),
        default="uniform",
        help=(
            "place COQM anchors uniformly or at train-set "
            "quartile-bin medians"
        ),
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--print_flops", action="store_true")
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.eval_mc_samples < 1:
        parser.error("--eval_mc_samples must be positive")
    if args.eval_policy == "mc_gumbel" and args.eval_mc_samples < 2:
        parser.error("--eval_policy mc_gumbel requires at least 2 samples")
    if args.eval_policy == "mc_gumbel" and not args.test:
        parser.error(
            "--eval_policy mc_gumbel is diagnostic and requires --test"
        )
    for name, value in (
            ("--shared_rank_weight", args.shared_rank_weight),
            ("--shared_rank_margin", args.shared_rank_margin),
            ("--shared_list_weight", args.shared_list_weight),
            ("--shared_list_margin", args.shared_list_margin),
            ("--ordinal_prototype_weight",
             args.ordinal_prototype_weight),
            ("--ordinal_prototype_order_margin",
             args.ordinal_prototype_order_margin),
            ("--ordinal_prototype_order_scale",
             args.ordinal_prototype_order_scale),
            ("--coqm_weight", args.coqm_weight),
            ("--coqm_coord_scale", args.coqm_coord_scale)):
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and non-negative")
    if args.ordinal_prototype_dim < 1:
        parser.error("--ordinal_prototype_dim must be positive")
    if args.coqm_dim < 2:
        parser.error("--coqm_dim must be at least two")
    if (
            not math.isfinite(args.shared_rank_temperature)
            or args.shared_rank_temperature <= 0):
        parser.error(
            "--shared_rank_temperature must be finite and positive"
        )
    if (
            not math.isfinite(args.shared_list_temperature)
            or args.shared_list_temperature <= 0):
        parser.error(
            "--shared_list_temperature must be finite and positive"
        )
    if (
            not math.isfinite(args.ordinal_prototype_temperature)
            or args.ordinal_prototype_temperature <= 0):
        parser.error(
            "--ordinal_prototype_temperature must be finite and positive"
        )
    if (
            not math.isfinite(args.coqm_temperature)
            or args.coqm_temperature <= 0):
        parser.error("--coqm_temperature must be finite and positive")
    if (
            not math.isfinite(args.coqm_theta_max)
            or args.coqm_theta_max <= 0
            or args.coqm_theta_max > math.pi / 2):
        parser.error("--coqm_theta_max must be in (0, pi / 2]")
    active_objectives = sum(
        value > 0
        for value in (
            args.shared_rank_weight,
            args.shared_list_weight,
            args.ordinal_prototype_weight,
            args.coqm_weight,
        )
    )
    if active_objectives > 1:
        parser.error(
            "pairwise, List-wise, prototype, and COQM supervision are "
            "mutually exclusive"
        )
    if (
            active_objectives > 0
            and not args.multi_modality):
        parser.error(
            "auxiliary supervision requires --multi_modality"
        )
    if args.test and args.print_flops:
        parser.error("--test and --print_flops are mutually exclusive")


def load_config(action, feats):
    if action in ("Ball", "Clubs", "Hoop", "Ribbon"):
        if feats == 1:
            import configs.RG_feats1 as config
        else:
            import configs.RG_feats2 as config
    else:
        if feats == 1:
            import configs.FISV_feats1 as config
        else:
            import configs.FISV_feats2 as config
    return config


def build_configs(args, config):
    current = (
        config.multi_modality
        if args.multi_modality
        else config.single_modality
    )
    epoch_num = (
        current["epoch"][args.action]
        if args.epochs is None else args.epochs
    )
    dataset_config = {
        "dataset_name": args.action,
        "feat_dir": args.feat_dir,
        "rgb_feat": config.features["V"],
        "flow_feat": config.features["F"],
        "audio_feat": config.features["A"],
        "batch_size": current["batch_size"],
        "num_workers": current["num_workers"],
    }
    ordinal_thresholds = (
        get_train_quality_thresholds(args.action)
        if args.ordinal_prototype_weight > 0 else ()
    )
    ordinal_centers = (
        get_train_quality_centers(args.action, ordinal_thresholds)
        if (
            args.ordinal_prototype_weight > 0
            and args.ordinal_prototype_target_mode == "continuous"
        ) else ()
    )
    coqm_anchor_scores = (
        get_train_quality_centers(args.action)
        if (
            args.coqm_weight > 0
            and args.coqm_anchor_mode == "train_centers"
        ) else ()
    )

    if args.multi_modality:
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
                "shared_rank_weight": args.shared_rank_weight,
                "shared_rank_margin": args.shared_rank_margin,
                "shared_rank_temperature":
                    args.shared_rank_temperature,
                "shared_list_weight": args.shared_list_weight,
                "shared_list_margin": args.shared_list_margin,
                "shared_list_temperature":
                    args.shared_list_temperature,
                "ordinal_prototype_weight":
                    args.ordinal_prototype_weight,
                "ordinal_quality_thresholds": (
                    ordinal_thresholds
                ),
                "ordinal_prototype_dim":
                    args.ordinal_prototype_dim,
                "ordinal_prototype_temperature":
                    args.ordinal_prototype_temperature,
                "ordinal_prototype_order_margin":
                    args.ordinal_prototype_order_margin,
                "ordinal_prototype_order_scale":
                    args.ordinal_prototype_order_scale,
                "ordinal_prototype_target_mode":
                    args.ordinal_prototype_target_mode,
                "ordinal_grade_centers": ordinal_centers,
                "quality_aux_rng_isolated":
                    args.quality_aux_rng_isolated,
                "coqm_weight": args.coqm_weight,
                "coqm_dim": args.coqm_dim,
                "coqm_temperature": args.coqm_temperature,
                "coqm_theta_max": args.coqm_theta_max,
                "coqm_coord_scale": args.coqm_coord_scale,
                "coqm_anchor_scores": coqm_anchor_scores,
                "eval_policy": args.eval_policy,
            },
        }
    else:
        model_config = {
            "model_name": "pamfn_base",
            "model_conf": {
                "in_dim": current["in_dim"][args.modality],
                "model_dim": current["model_dim"],
                "drop_rate": current["drop_rate"],
                "modality": args.modality,
            },
        }

    optimizer_config = {
        "optim": current["optim"],
        "lr": current["lr"],
        "weight_decay": current["weight_decay"],
        "reduce_fc_lr": current["reduce_fc_lr"],
    }
    scheduler_config = {
        "scheduler": current["lr_scheduler"],
        "epoch_num": epoch_num,
        "scheduler_steps": current["epoch"],
        "lr_min": current["lr_min"],
        "warmup_lr_init": current["warmup_lr_init"],
        "decay_rate": 0.1,
    }
    return (
        current,
        epoch_num,
        dataset_config,
        model_config,
        optimizer_config,
        scheduler_config,
    )


def log_configuration(log_func, args, epoch_num):
    log_func(
        f"Run: action={args.action} feats={args.feats} "
        f"seed={args.seed} epochs={epoch_num}"
    )
    log_func(
        "Evaluation protocol: official full train/test; original legacy "
        "crop; test every epoch; "
        f"routing={args.eval_policy}; mc_samples={args.eval_mc_samples}; "
        f"eval_seed={args.eval_seed}"
    )
    if args.shared_rank_weight > 0:
        log_func(
            "Training objective: official MSE + "
            f"{args.shared_rank_weight:g}*pairwise logistic; "
            f"label_gap>{args.shared_rank_margin:g}; "
            f"temperature={args.shared_rank_temperature:g}"
        )
    elif args.shared_list_weight > 0:
        log_func(
            "Training objective: official MSE + "
            f"{args.shared_list_weight:g}*grouped ListMLE; "
            f"tie_range<={args.shared_list_margin:g}; "
            f"temperature={args.shared_list_temperature:g}"
        )
    elif args.ordinal_prototype_weight > 0:
        thresholds = get_train_quality_thresholds(args.action)
        log_func(
            "Training objective: official MSE + "
            f"{args.ordinal_prototype_weight:g}*ordinal prototypes; "
            f"dim={args.ordinal_prototype_dim}; "
            f"temperature={args.ordinal_prototype_temperature:g}; "
            f"order_margin={args.ordinal_prototype_order_margin:g}; "
            f"order_scale={args.ordinal_prototype_order_scale:g}; "
            f"target={args.ordinal_prototype_target_mode}; "
            f"rng_isolated={args.quality_aux_rng_isolated}"
        )
        log_func(
            "Train-only quality quartiles: "
            + ", ".join(f"{value:.6f}" for value in thresholds)
        )
        if args.ordinal_prototype_target_mode == "continuous":
            centers = get_train_quality_centers(
                args.action, thresholds
            )
            log_func(
                "Train-only grade centers: "
                + ", ".join(f"{value:.6f}" for value in centers)
            )
    elif args.coqm_weight > 0:
        anchor_scores = (
            get_train_quality_centers(args.action)
            if args.coqm_anchor_mode == "train_centers"
            else (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
        )
        log_func(
            "Training objective: official MSE + "
            f"{args.coqm_weight:g}*COQM; dim={args.coqm_dim}; "
            f"anchors=4; temperature={args.coqm_temperature:g}; "
            f"theta_max={args.coqm_theta_max:g}; "
            f"coord_scale={args.coqm_coord_scale:g}; "
            f"anchor_mode={args.coqm_anchor_mode}; "
            "rng_isolated=True"
        )
        log_func(
            "COQM anchor scores: "
            + ", ".join(f"{value:.6f}" for value in anchor_scores)
        )
    else:
        log_func("Training objective: official PAMFN MSE")


def main():
    args = parser_args()
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.action, args.feats)
    (
        current,
        epoch_num,
        dataset_config,
        model_config,
        optimizer_config,
        scheduler_config,
    ) = build_configs(args, config)
    log_func = partial(log_info, logger=None, print_log=True)
    log_configuration(log_func, args, epoch_num)

    if args.print_flops:
        macs, params = print_flops(model_config)
        print(macs)
        print(params)
        return

    if args.test:
        checkpoint = args.ckpt_path or os.path.join(
            current["ckpt_dir"],
            f"{args.action}_multimodal.pth",
        )
        result = test_model(
            model_kwargs=model_config,
            dataset_kwargs=dataset_config,
            ckpt_path=checkpoint,
            log_func=log_func,
            seed=args.seed,
            eval_policy=args.eval_policy,
            eval_mc_samples=args.eval_mc_samples,
            eval_seed=args.eval_seed,
        )
        result["run"] = _run_metadata(args, epoch_num)
        _write_result_json(args.result_json, result)
        return

    results = go(
        model_config,
        optimizer_config,
        scheduler_config,
        dataset_config,
        epoch_num,
        save_base_model=args.save_base_model,
        save_moniter=current["save_moniter"],
        save_final_model=args.save_final_model,
        log_func=log_func,
        seed=args.seed,
        clip_grad=current["clip_grad"],
        eval_policy=args.eval_policy,
        eval_mc_samples=args.eval_mc_samples,
        eval_seed=args.eval_seed,
    )
    _write_result_json(args.result_json, {
        "best_avg_srcc": results[0] / 100,
        "best_srcc": results[1] / 100,
        "min_avg_mse": results[2] / 100,
        "min_mse": results[3] / 100,
        "saved_model_srcc": results[4] / 100,
        "run": _run_metadata(args, epoch_num),
    })
    print("%.4f\t%.4f\t%.4f\t%.4f" % results[:4])


def _write_result_json(path, result):
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


def _run_metadata(args, epoch_num):
    return {
        "tag": args.run_tag,
        "action": args.action,
        "seed": args.seed,
        "feats": args.feats,
        "epochs": epoch_num,
        "ordinal_prototype_weight": args.ordinal_prototype_weight,
        "ordinal_prototype_target_mode":
            args.ordinal_prototype_target_mode,
        "coqm_weight": args.coqm_weight,
        "coqm_coord_scale": args.coqm_coord_scale,
        "coqm_anchor_mode": args.coqm_anchor_mode,
        "coqm_anchor_scores": (
            list(get_train_quality_centers(args.action))
            if (
                args.coqm_weight > 0
                and args.coqm_anchor_mode == "train_centers"
            ) else (
                [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
                if args.coqm_weight > 0 else []
            )
        ),
        "quality_aux_rng_isolated": args.quality_aux_rng_isolated,
        "eval_policy": args.eval_policy,
        "eval_mc_samples": args.eval_mc_samples,
        "eval_seed": args.eval_seed,
    }


if __name__ == "__main__":
    main()
