import argparse
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import configs.FISV_feats2 as config  # noqa: E402
from models.pamfn import PAMFN  # noqa: E402
from pipeline import load_checkpoint, state_dict_digest  # noqa: E402


def model_kwargs(**overrides):
    current = config.multi_modality
    values = {
        "model_dim": current["model_dim"],
        "fc_drop": current["fc_drop"],
        "fc_r": current["fc_r"],
        "feat_drop": current["feat_drop"],
        "K": current["K"],
        "ms_heads": current["ms_heads"],
        "cm_heads": current["cm_heads"],
        "ckpt_dir": os.path.join(
            ROOT, "pretrained_models", "feats2"
        ),
        "rgb_ckpt_name": current["rgb_ckpt_name"],
        "flow_ckpt_name": current["flow_ckpt_name"],
        "audio_ckpt_name": current["audio_ckpt_name"],
        "dataset_name": "TES",
    }
    values.update(overrides)
    return values


def model_input(batch_size=2):
    torch.manual_seed(37)
    return {
        "V": torch.randn(batch_size, 16, 1024),
        "F": torch.randn(batch_size, 16, 1024),
        "A": torch.randn(batch_size, 16, 768),
    }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            ROOT, "pretrained_models", "feats2", "TES_multimodal.pth"
        ),
    )
    parser.add_argument(
        "--branch_ckpt_dir",
        default=os.path.join(ROOT, "pretrained_models", "feats2"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    checkpoint = os.path.abspath(args.checkpoint)
    branch_ckpt_dir = os.path.abspath(args.branch_ckpt_dir)

    base = PAMFN(**model_kwargs(ckpt_dir=branch_ckpt_dir))
    stage_motion = PAMFN(**model_kwargs(
        ckpt_dir=branch_ckpt_dir,
        motion_saliency_proxy=True,
        motion_saliency_alpha=0.015,
        motion_saliency_source="delta",
        motion_saliency_positive_only=True,
    ))
    load_checkpoint(base, checkpoint)
    load_checkpoint(stage_motion, checkpoint)

    assert parameter_count(stage_motion) == parameter_count(base)
    assert not any(
        name.startswith((
            "motion_encoder.", "motion_score_residual_head."
        ))
        for name in stage_motion.state_dict()
    )

    base.eval()
    stage_motion.eval()
    inputs = model_input()
    stage_calls = []
    original_apply = stage_motion._apply_motion_residual

    def traced_apply(feat, motion_source, normalized_gate):
        output, correction = original_apply(
            feat, motion_source, normalized_gate
        )
        stage_calls.append({
            "target_length": feat.shape[2],
            "gate": float(normalized_gate.mean()),
            "correction_max": float(correction.abs().max()),
        })
        return output, correction

    stage_motion._apply_motion_residual = traced_apply
    digest_before = state_dict_digest(stage_motion)
    with torch.no_grad():
        base_prediction, _ = base(
            inputs, force_gates=(0.0, 0.0)
        )
        motion_off, off_info = stage_motion(
            inputs, force_gates=(0.0, 0.0)
        )
        motion_on, on_info = stage_motion(
            inputs, force_gates=(1.0, 0.0)
        )
    digest_after = state_dict_digest(stage_motion)

    assert torch.equal(motion_off, base_prediction)
    assert not torch.equal(motion_on, motion_off)
    assert float((motion_on - motion_off).abs().max()) > 0
    assert float(off_info["motion_gate"]) == 0.0
    assert float(on_info["motion_gate"]) == 1.0
    assert float(on_info["motion_correction"]) > 0
    assert torch.count_nonzero(on_info["motion_score_correction"]) == 0
    assert torch.count_nonzero(on_info["skeleton_score_correction"]) == 0
    off_calls = [call for call in stage_calls if call["gate"] == 0.0]
    on_calls = [call for call in stage_calls if call["gate"] == 1.0]
    assert len(off_calls) == 2
    assert len(on_calls) == 2
    assert len({call["target_length"] for call in on_calls}) == 2
    assert all(call["correction_max"] == 0.0 for call in off_calls)
    assert all(call["correction_max"] > 0.0 for call in on_calls)
    assert digest_after == digest_before

    print("stage Motion uses zero fresh parameters: passed")
    print("gate0 equals the strict-loaded cue-free Base: passed")
    print("gate1 changes the score through stage1/stage2 only: passed")
    print("inference leaves the complete Base state unchanged: passed")
    print("stage Motion probe verification passed")


if __name__ == "__main__":
    main()
