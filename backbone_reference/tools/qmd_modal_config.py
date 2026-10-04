"""v16: preserve Ordered, compare information source, factorization and KD."""
FUSED='ordered_modal_fused_gt'
CONCAT='ordered_modal_concat_gt'
EXPERT='ordered_modal_expert_gt'
KD='ordered_modal_expert_kd'
RECIPES={FUSED:dict(features='fused',heads='joint',rank=0.,weight=0.),
    CONCAT:dict(features='modal',heads='joint',rank=0.,weight=0.),
    EXPERT:dict(features='modal',heads='separate',rank=0.,weight=0.),
    KD:dict(features='modal',heads='separate',rank=0.,weight=.5)}
MODALITIES=('V','F','A')
V13_GT='ordered_bc_mlp16_gt'
V13_KD='ordered_bc_mlp16_kd_w050'
V15_GATE='ordered_pair_gate'
LR=.0001
TARGET_RATIO=.075
RANK_RATIO=0.
BETA_CAP=10000.
LABEL_GAP=.02


def branch_config(tag,input_dim=768):
    if tag not in RECIPES or input_dim%3:raise ValueError('invalid modal recipe/dimension')
    r=RECIPES[tag]
    return dict(input_dim=input_dim,hidden=16,bound=.03,feature_clip=5.,std_floor=.01,
        features=r['features'],heads=r['heads'],modalities=list(MODALITIES),
        capture='fused fc input and unimodal feats[-1]; same Ordered forward',
        combination='mean of three bounded residuals' if r['heads']=='separate' else 'one bounded residual')


def parameter_count(config):return config['input_dim']*16+(96 if config['heads']=='separate' else 32)


def calibration_contract():
    return dict(policy='per_batch_detached_pair_norm',reference='centered GT plus mean penalty only',
        target_ratio=TARGET_RATIO,rank_ratio=0.,beta_cap=BETA_CAP,gradient_epsilon=1e-12,
        zero_gradient='skip current batch and recalibrate next batch')


def metadata(action,seed,tag,epochs,smoke=False):
    return dict(action=action,seed=seed,tag=tag,epochs=epochs,feats=2,
        protocol='july28_independent_modality_correction',recipe=RECIPES[tag],branch=branch_config(tag),
        anchor='July28 Ordered frozen checkpoint',eval_policy='fixed_ordered_eval_rng_float64_sum',
        calibration='one training-only pass; mean label-minus-anchor; selected feature mean/std',
        gt='batch-centered score MSE plus aggregate residual-mean penalty',mean_penalty=.1,
        expert_supervision='joint aggregate GT; heads have separate parameters and modality inputs, not independent teachers',
        reliability_fusion=False,out_of_fold=False,
        kd_target='stop-gradient full aggregate residual pair difference',
        gate=dict(label_gap=LABEL_GAP,improvement_epsilon=1e-6,
            rule='full corrected pair correctly ordered AND label-gap error improved over Ordered',
            inference='no labels or gate; fixed expert average'),
        kd_calibration=calibration_contract(),mask_ratio=.15,warmup_epochs=0,
        ramp_epochs=1 if smoke else 5,lr=LR,weight_decay=.01,epoch_zero_eligible=False)
