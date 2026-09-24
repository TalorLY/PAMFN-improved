"""v17: retain E and train only a zero-initialized, bounded modality supplement."""
FIXED='e_modal_fixed'
SCALED='e_modal_scaled'
SHRINK='e_modal_shrink'
RECIPES={FIXED:dict(relative=False,shrink=False),SCALED:dict(relative=True,shrink=False),
         SHRINK:dict(relative=True,shrink=True)}
ABS_BOUND=.003
RELATIVE_RATIO=.1
LR=.0001
GATE_LR=.001
GATE_PENALTY=1.
SEEDS=(0,1,2)
E_TAG='ordered_modal_fused_gt'
REFERENCE_E='ordered_bc_mlp16_gt'


def branch_config(tag,input_dim=768):
    if tag not in RECIPES or input_dim%3:raise ValueError('invalid supplement recipe/dimension')
    return dict(input_dim=input_dim,hidden=16,modalities=['V','F','A'],abs_bound=ABS_BOUND,
        relative_ratio=RELATIVE_RATIO,recipe=RECIPES[tag],std_floor=.01,feature_clip=5.,
        combination='fixed mean of three heads times global gate',gate_initial=1.)


def parameter_count(tag,input_dim=768):return input_dim*16+96+int(RECIPES[tag]['shrink'])


def metadata(action,seed,tag,epochs,smoke=False):
    return dict(action=action,seed=seed,tag=tag,epochs=epochs,feats=2,smoke=smoke,
        protocol='frozen_E_plus_modal_supplement',branch=branch_config(tag),recipe=RECIPES[tag],
        anchor='frozen July28 Ordered plus frozen v16 FusedGT (exact v13 E replay)',
        calibration='one training-only pass; modal feature mean/std; frozen E prediction std',
        bound='0.003' if tag==FIXED else 'min(0.003, 0.1 * training E prediction std)',
        gt='batch-centered corrected-score MSE plus 0.1 * supplement mean squared',
        shrink_penalty='bound squared times projected scalar gate' if tag==SHRINK else 'none',
        gate_lr=GATE_LR if tag==SHRINK else 0.,gate_penalty=GATE_PENALTY if tag==SHRINK else 0.,
        gate_projection='[0,1]; may reach exact zero; fixed 1 in controls',
        lr=LR,weight_decay=.01,kd_weight=0.,rank_weight=0.,masked_forward=False,
        eval_policy='fixed Ordered RNG; preserve E double addition order; append supplement in double',
        selection='trained-epoch test-best; no epoch-zero or per-action fallback',
        epoch_zero_eligible=False,out_of_fold=False,sample_reliability_router=False)
