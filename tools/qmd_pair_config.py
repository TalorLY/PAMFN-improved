"""v15: frozen Ordered, supervised pair repair and useful-teacher pair KD."""
GT='ordered_pair_gt'
RANK='ordered_pair_rank'
GATE='ordered_pair_gate'
COMBO='ordered_pair_rank_gate'
RECIPES={
    GT:dict(rank=0.,weight=0.),
    RANK:dict(rank=1.,weight=0.),
    GATE:dict(rank=0.,weight=.5),
    COMBO:dict(rank=1.,weight=.5),
}
CANDIDATES=(RANK,GATE,COMBO)
V13_GT='ordered_bc_mlp16_gt'
V13_KD='ordered_bc_mlp16_kd_w050'
LR=.0001
RAMP_EPOCHS=5
TARGET_RATIO=.075
RANK_RATIO=.1
BETA_CAP=10000.
GRAD_EPS=1e-12
LABEL_GAP=.02
RESIDUAL_BOUND=.03
MARGIN_SCALE=.1
MARGIN_CAP=.02
PROTECT_WEIGHT=.25
GATE_EPS=1e-6


def branch_config(tag,input_dim=768):
    if tag not in RECIPES:raise ValueError('unknown v15 recipe')
    return dict(input_dim=input_dim,hidden=16,bound=RESIDUAL_BOUND,feature_clip=5.,std_floor=.01)


def parameter_count(config):return config['input_dim']*16+32


def calibration_contract():
    return dict(policy='per_batch_detached_pair_norm',reference='centered GT plus mean penalty only',
        target_ratio=TARGET_RATIO,rank_ratio=RANK_RATIO,beta_cap=BETA_CAP,gradient_epsilon=GRAD_EPS,
        zero_gradient='skip current batch and recalibrate next batch')


def metadata(action,seed,tag,epochs,smoke=False):
    return dict(action=action,seed=seed,tag=tag,epochs=epochs,feats=2,
        protocol='july28_selective_pair_correction',recipe=RECIPES[tag],branch=branch_config(tag),
        anchor='July28 Ordered frozen checkpoint',eval_policy='fixed_ordered_eval_rng_float64_sum',
        calibration='one training-only pass; mean label-minus-anchor; feature mean/std',
        gt='batch-centered score MSE plus residual-mean penalty',mean_penalty=.1,
        pair_supervision=dict(label_gap=LABEL_GAP,margin_scale=MARGIN_SCALE,margin_cap=MARGIN_CAP,
            protect_weight=PROTECT_WEIGHT,loss='mean squared hinge per anchor-hard/protected group',
            selection='training labels only; repair anchor-hard pairs only if margin deficit < 2*residual_bound - gate_epsilon'),
        kd_target='stop-gradient full residual pair difference',
        gate=dict(label_gap=LABEL_GAP,improvement_epsilon=GATE_EPS,
            rule='full corrected pair correctly ordered AND absolute label-gap error improved over anchor+bias',
            inference='no labels or gate; always use learned bounded residual'),
        kd_calibration=calibration_contract(),mask_ratio=.15,warmup_epochs=0,
        ramp_epochs=1 if smoke else RAMP_EPOCHS,lr=LR,weight_decay=.01,epoch_zero_eligible=False)
