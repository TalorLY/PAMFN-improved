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
    parser.add_argument('--score_head', choices=('original', 'linear', 'groupnorm'), default='original')
    parser.add_argument('--score_head_diagnostics', action='store_true')
    parser.add_argument('--ema_screen', action='store_true')
    parser.add_argument('--ema_view', choices=('online', 'online_bn', 'ema_short', 'ema_long'), default='online')
    parser.add_argument('--qmd_mode', choices=('off', 'mask_gt', 'score_qmd', 'score_rank_qmd', 'reliable_score_qmd'), default='off')
    parser.add_argument('--qmd_weight', type=float, default=1.0)
    parser.add_argument('--qmd_grad_mode', choices=('off', 'observe', 'kd', 'all'), default='off')
    parser.add_argument('--qmd_score_gate_floor', type=float, default=0.0)
    parser.add_argument('--qmd_mask_ratio', type=float, default=0.15)
    parser.add_argument('--qmd_gt_weight', type=float, default=0.5)
    parser.add_argument('--qmd_target_ratio', type=float, default=0.075)
    parser.add_argument('--qmd_beta_cap', type=float, default=1.0)
    parser.add_argument('--qmd_calibration_mode', choices=('single', 'shared', 'modality'), default='single')
    parser.add_argument('--qmd_calibration_batches', type=int, default=9)
    parser.add_argument('--qmd_warmup', type=float, default=0.1)
    parser.add_argument('--qmd_rampup', type=float, default=0.1)
    parser.add_argument('--qmd_rank_weight', type=float, default=0.5)
    parser.add_argument('--qmd_rank_gap', type=float, default=0.05)
    parser.add_argument('--qmd_rank_gamma', type=float, default=0.1)
    parser.add_argument("--print_flops", action="store_true")
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    if args.ema_view != 'online' and (not args.test or not args.ema_screen):
        parser.error('--ema_view is only for --test --ema_screen checkpoint reloads')
    if args.ema_screen:
        if (not args.multi_modality or args.score_head != 'original' or args.eval_policy != 'argmax'
                or args.ordinal_prototype_weight or args.coqm_weight or args.qmd_mode != 'score_qmd'
                or args.shared_rank_weight or args.shared_list_weight or args.qmd_calibration_mode != 'single'):
            parser.error('EMA screen requires original Score-QMD, original head, argmax and no Ordered/COQM')
        if not args.test and not args.save_final_model:
            parser.error('--ema_screen requires --save_final_model')
    if (args.score_head != 'original' or args.score_head_diagnostics) and not args.multi_modality:
        parser.error('score head experiments require --multi_modality')
    if args.qmd_calibration_batches < 6 or args.qmd_calibration_batches % 3:
        parser.error('--qmd_calibration_batches must be a multiple of 3 and at least 6')
    if args.qmd_calibration_mode != 'single' and args.qmd_mode != 'score_qmd':
        parser.error('window calibration requires --qmd_mode score_qmd')
    if not math.isfinite(args.qmd_score_gate_floor) or not 0 <= args.qmd_score_gate_floor <= 1:
        parser.error('--qmd_score_gate_floor must be in [0, 1]')
    if args.qmd_score_gate_floor and args.qmd_mode != 'reliable_score_qmd':
        parser.error('--qmd_score_gate_floor requires reliable_score_qmd')
    for key in ('qmd_weight', 'qmd_gt_weight', 'qmd_target_ratio', 'qmd_beta_cap',
                'qmd_warmup', 'qmd_rampup', 'qmd_rank_weight', 'qmd_rank_gap', 'qmd_rank_gamma'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            parser.error('--' + key + ' must be finite and nonnegative')
    if args.qmd_grad_mode != 'off' and (args.qmd_mode != 'score_qmd' or args.qmd_weight <= 0):
        parser.error('gradient guard requires active ordinary Score-QMD')
    if not 0 < args.qmd_mask_ratio < 1 or args.qmd_warmup + args.qmd_rampup > 1:
        parser.error('invalid QMD mask ratio or warmup/rampup')
    if args.qmd_rank_gamma <= 0:
        parser.error('--qmd_rank_gamma must be positive')
    if args.qmd_mode != 'off' and args.qmd_weight > 0:
        if not args.multi_modality:
            parser.error('QMD requires --multi_modality')
        if any(x > 0 for x in (args.coqm_weight, args.shared_rank_weight, args.shared_list_weight)):
            parser.error('QMD only supports the Ordered auxiliary combination')
        if args.ordinal_prototype_weight > 0 and not args.quality_aux_rng_isolated:
            parser.error('QMD + Ordered requires --quality_aux_rng_isolated')
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
    if args.qmd_mode != 'off' and args.qmd_weight > 0 and not args.test:
        dataset_config['return_valid_lengths'] = True
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
                "score_head": args.score_head,
                "score_head_diagnostics": args.score_head_diagnostics,
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
    if args.qmd_mode == 'reliable_score_qmd':
        log_func('Reliable Score-KD: distill only if abs(full-y) < abs(masked-y); '
                 'training labels only; ungated beta calibration; no Rank-KD')
    if args.qmd_mode != 'off':
        log_func('QMD config: ' + json.dumps(qmd_config(args), sort_keys=True))
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
        if args.qmd_mode != 'off' and args.qmd_weight > 0:
            log_func('Additional training objective: scheduled Mask-GT + Score-KD'
                     + (' + reliability-gated Rank-KD' if args.qmd_mode == 'score_rank_qmd' else ''))
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
        if args.qmd_mode != 'off' and args.qmd_weight > 0:
            log_func('Training objective: full-view MSE + scheduled Mask-GT'
                     + (' + calibrated Score-KD' if args.qmd_mode in ('score_qmd', 'score_rank_qmd', 'reliable_score_qmd') else '')
                     + (' + reliability-gated Rank-KD' if args.qmd_mode == 'score_rank_qmd' else '')
                     + (' + full-view Ordered' if args.ordinal_prototype_weight > 0 else ''))
        else:
            log_func("Training objective: official PAMFN MSE")


def main():
    args = parser_args()
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if args.qmd_mode != 'off':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('QMD requested GPU is unavailable; check GPU index/environment')
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
            return_predictions=args.qmd_grad_mode != 'off',
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
        qmd_config=(qmd_config(args) if args.qmd_mode != 'off' and args.qmd_weight > 0 else None),
        qmd_report_path=(os.path.join(os.path.dirname(args.result_json), 'qmd_training.json')
                         if args.result_json else ''),
        ema_screen=args.ema_screen,
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
    result = {
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
        "qmd": qmd_config(args),
        "ordered_config": (dict(dim=args.ordinal_prototype_dim,
                                 temperature=args.ordinal_prototype_temperature,
                                 margin=args.ordinal_prototype_order_margin,
                                 order_scale=args.ordinal_prototype_order_scale)
                           if args.ordinal_prototype_weight > 0 else None),
    }
    if args.score_head != 'original' or args.score_head_diagnostics:
        result['score_head'] = dict(variant=args.score_head, diagnostics=args.score_head_diagnostics)
    if args.ema_screen:
        from models.ema_shadow import recipe
        result['ema'] = dict(recipe=recipe(), view=args.ema_view)
    return result


def qmd_config(args):
    config = dict(mode=args.qmd_mode, weight=args.qmd_weight,
                mask_ratio=args.qmd_mask_ratio, mask_gt_weight=args.qmd_gt_weight,
                target_ratio=args.qmd_target_ratio, beta_cap=args.qmd_beta_cap,
                warmup=args.qmd_warmup, rampup=args.qmd_rampup)
    if args.qmd_mode == 'score_rank_qmd':
        config.update(rank_weight=args.qmd_rank_weight, rank_gap=args.qmd_rank_gap,
                      rank_gamma=args.qmd_rank_gamma)
    if args.qmd_mode == 'reliable_score_qmd':
        config['score_gate_floor'] = args.qmd_score_gate_floor
    if args.qmd_calibration_mode != 'single':
        config.update(calibration_mode=args.qmd_calibration_mode,
                      calibration_batches=args.qmd_calibration_batches)
    if getattr(args, 'qmd_grad_mode', 'off') != 'off':
        config['grad_mode'] = args.qmd_grad_mode
    return config


if __name__ == "__main__":
    main()
