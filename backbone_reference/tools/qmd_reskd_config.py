"""v14: 2 energy settings x 3 KD objectives; identical in both repositories."""
GT='ordered_rkd_gt'
SCORE='ordered_rkd_score'
RESIDUAL='ordered_rkd_residual'
ENERGY_GT='ordered_rkd_energy_gt'
ENERGY_SCORE='ordered_rkd_energy_score'
ENERGY_RESIDUAL='ordered_rkd_energy_residual'
RECIPES={
    GT:dict(energy=0.,target='score',weight=0.),
    SCORE:dict(energy=0.,target='score',weight=.5),
    RESIDUAL:dict(energy=0.,target='residual',weight=.5),
    ENERGY_GT:dict(energy=1.,target='score',weight=0.),
    ENERGY_SCORE:dict(energy=1.,target='score',weight=.5),
    ENERGY_RESIDUAL:dict(energy=1.,target='residual',weight=.5),
}
V13_GT='ordered_bc_mlp16_gt'
V13_KD='ordered_bc_mlp16_kd_w050'
LR=.0001
RAMP_EPOCHS=5
TARGET_RATIO=.075
BETA_CAP=10000.
GRAD_EPS=1e-12


def branch_config(tag,input_dim=768):
    if tag not in RECIPES:raise ValueError('unknown v14 recipe')
    return dict(input_dim=input_dim,hidden=16,bound=.03,feature_clip=5.,std_floor=.01)


def parameter_count(config):return config['input_dim']*16+32


def calibration_contract():
    return dict(policy='per_batch_detached_norm',reference='centered GT plus mean penalty, excluding energy',
        target_ratio=TARGET_RATIO,beta_cap=BETA_CAP,gradient_epsilon=GRAD_EPS,
        zero_gradient='skip current batch and recalibrate next batch')


def metadata(action,seed,tag,epochs,smoke=False):
    return dict(action=action,seed=seed,tag=tag,epochs=epochs,feats=2,
        protocol='july28_residual_space_kd',recipe=RECIPES[tag],branch=branch_config(tag),
        anchor='July28 Ordered frozen checkpoint',eval_policy='fixed_ordered_eval_rng_float64_sum',
        calibration='one training-only pass; mean label-minus-anchor; feature mean/std',
        gt='batch-centered score MSE plus residual-mean penalty',mean_penalty=.1,
        energy='mean squared full-input residual; coefficient fixed in recipe',
        kd_calibration=calibration_contract(),mask_ratio=.15,warmup_epochs=0,
        ramp_epochs=1 if smoke else RAMP_EPOCHS,lr=LR,weight_decay=.01,epoch_zero_eligible=False)
