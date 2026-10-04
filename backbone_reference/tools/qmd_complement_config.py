"""Fixed v13 experiment contract; identical in coordinator and trainer."""
RECIPES = {
    'ordered_bc_linear_gt': dict(hidden=0, target='corrected', weight=0.),
    'ordered_bc_linear_raw_kd': dict(hidden=0, target='raw', weight=.5),
    'ordered_bc_linear_kd_w050': dict(hidden=0, target='corrected', weight=.5),
    'ordered_bc_linear_kd_w100': dict(hidden=0, target='corrected', weight=1.),
    'ordered_bc_mlp16_gt': dict(hidden=16, target='corrected', weight=0.),
    'ordered_bc_mlp16_kd_w050': dict(hidden=16, target='corrected', weight=.5),
}
LINEAR_GT = 'ordered_bc_linear_gt'
MLP_GT = 'ordered_bc_mlp16_gt'
BOUND = .03
LR = .0001
MEAN_PENALTY = .1
RAMP_EPOCHS = 5

def branch_config(tag, input_dim=768):
    if tag not in RECIPES: raise ValueError('unknown v13 recipe')
    return dict(input_dim=input_dim, hidden=RECIPES[tag]['hidden'], bound=BOUND,
                feature_clip=5., std_floor=.01)

def parameter_count(config):
    dim, hidden = config['input_dim'], config['hidden']
    return dim if hidden == 0 else dim*hidden+2*hidden

def metadata(action, seed, tag, epochs, smoke=False):
    return dict(action=action, seed=seed, tag=tag, epochs=epochs, feats=2,
        protocol='july28_bias_separated', recipe=RECIPES[tag], branch=branch_config(tag),
        anchor='July28 Ordered frozen checkpoint', eval_policy='fixed_ordered_eval_rng_float64_sum',
        calibration='one training-only pass; mean label-minus-anchor; feature mean/std',
        gt='batch-centered score MSE plus residual-mean penalty', mean_penalty=MEAN_PENALTY,
        mask_ratio=.15, warmup_epochs=0, ramp_epochs=1 if smoke else RAMP_EPOCHS,
        target_ratio=.075, beta_cap=1., lr=LR, weight_decay=.01, epoch_zero_eligible=False)
