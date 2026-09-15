"""Fixed-window, training-only score-head gradient calibration (no model state)."""
import math


class WindowCalibration:
    """Collect balanced masked batches, then freeze shared or modality betas.

    The collection window uses Mask-GT but no KD. Both candidates therefore
    see identical updates until the window ends. KD begins on the NEXT batch.
    Ratio-of-sample-weighted-mean norms, not mean of potentially huge ratios.
    This is a norm proxy over the window, not a guarantee on later gradients.
    """
    def __init__(self, mode, batches, target, cap, weight):
        if mode not in ('shared', 'modality'):
            raise ValueError('unknown window calibration mode')
        if not isinstance(batches, int) or batches < 6 or batches % 3:
            raise ValueError('calibration batches must be a multiple of 3 and at least 6')
        if any(not math.isfinite(x) or x < 0 for x in (target, cap, weight)):
            raise ValueError('invalid calibration weights')
        self.mode, self.batches = mode, batches
        self.target, self.cap, self.weight = target, cap, weight
        self.observations = []
        self.report = None

    def observe(self, modality, main_norm, kd_norm, samples, epoch):
        if self.report is not None:
            raise RuntimeError('calibration is already frozen')
        if modality not in ('V', 'F', 'A') or samples < 1:
            raise ValueError('invalid calibration batch')
        if any(not math.isfinite(x) or x < 0 for x in (main_norm, kd_norm)):
            raise FloatingPointError('non-finite/negative QMD calibration gradients')
        self.observations.append(dict(modality=modality, main_gradient_norm=main_norm,
            kd_gradient_norm=kd_norm, samples=samples, epoch=epoch + 1))
        if len(self.observations) < self.batches:
            return
        counts = {m: sum(r['modality'] == m for r in self.observations) for m in ('V', 'F', 'A')}
        if set(counts.values()) != {self.batches // 3}:
            raise ValueError('calibration window must contain equal modality batch counts')
        groups = {}
        for key in (('shared',) if self.mode == 'shared' else ('V', 'F', 'A')):
            rows = [r for r in self.observations if key == 'shared' or r['modality'] == key]
            n = sum(r['samples'] for r in rows)
            main = sum(r['main_gradient_norm'] * r['samples'] for r in rows) / n
            kd = sum(r['kd_gradient_norm'] * r['samples'] for r in rows) / n
            degenerate = main <= 1e-12 or kd <= 1e-12
            raw = 0.0 if degenerate else self.target * main / kd
            if not math.isfinite(raw):
                raise FloatingPointError('non-finite QMD beta')
            beta = min(self.cap, raw)
            groups[key] = dict(batches=len(rows), samples=n, main_gradient_norm=main,
                kd_gradient_norm=kd, uncapped_beta=raw, beta=beta, capped=raw > self.cap,
                degenerate=degenerate, achieved_ratio_at_full_ramp=(
                    self.weight * beta * kd / main if not degenerate else 0.0))
        self.report = dict(mode=self.mode, frozen=True, batches=self.batches,
            modality_counts=counts, start_epoch=self.observations[0]['epoch'], epoch=epoch + 1,
            target_ratio=self.target, beta_cap=self.cap, weight=self.weight,
            beta={k: v['beta'] for k, v in groups.items()}, groups=groups,
            degenerate=any(g['degenerate'] for g in groups.values()),
            scope='score_head_parameters_first_balanced_training_window',
            estimator='ratio_of_sample_weighted_mean_gradient_norms',
            collection_objective='main_plus_ramped_mask_gt_no_kd',
            kd_starts_active_batch=self.batches + 1, observations=self.observations)

    def beta_for(self, modality):
        if self.report is None:
            return 0.0
        return self.report['beta']['shared' if self.mode == 'shared' else modality]
