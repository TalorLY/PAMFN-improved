"""One-sided auxiliary gradient projection; no model state, RNG or optimizer changes.

Inspired by PCGrad (Yu et al., NeurIPS 2020), but the original full-view objective
is never projected. Euclidean first-order alignment is NOT an Adam/SRCC guarantee.
"""
import math
import warnings
import torch

MODES = ('off', 'observe', 'kd', 'all')


def recipe(mode):
    if mode not in MODES:
        raise ValueError('unknown gradient guard mode')
    return dict(mode=mode, scope='trainable_parameters_shared_with_main_objective',
                anchor='unchanged_full_view_objective_including_Ordered',
                target='weighted_kd' if mode == 'kd' else 'weighted_mask_gt_plus_kd',
                projection='one_sided_euclidean_negative_dot_only',
                optimizer='unchanged', normalization='global_shared_vector',
                epsilon=1e-24, inference='unchanged')


def geometry(main, auxiliary):
    pairs = [(m, a) for m, a in zip(main, auxiliary) if m is not None and a is not None]
    if not pairs:
        return dict(dot=0., main_sq=0., aux_sq=0., cosine=0., shared=0)
    moments = []
    for m, a in pairs:
        # Bound temporary float64 memory to one parameter tensor at a time.
        m, a = m.detach().double(), a.detach().double()
        moments.append(torch.stack(((m*a).sum(),(m*m).sum(),(a*a).sum())))
    dot, mm, aa = torch.stack(moments).sum(0).tolist()
    if not all(math.isfinite(v) for v in (dot, mm, aa)):
        raise FloatingPointError('nonfinite gradient geometry')
    cosine = dot / math.sqrt(mm*aa) if mm > 1e-24 and aa > 1e-24 else 0.
    return dict(dot=dot, main_sq=mm, aux_sq=aa, cosine=max(-1., min(1., cosine)), shared=len(pairs))


def add_gradients(left, right):
    return tuple(b if a is None else a if b is None else a+b for a,b in zip(left,right))


def check_projection(before, after, main, target, coefficient):
    """Check only: never change the correction or any successful v8 update."""
    legacy_tolerance = 1e-5*max(abs(before['dot']), 1e-24)
    if not coefficient or after['dot'] >= -legacy_tolerance:
        return
    # Dot products are accumulated in float64, but m*c and a+m*c are rounded
    # in the gradient dtype. A nearly orthogonal dot is an unsuitable error
    # scale: use ||m|| (||a|| + |c| ||m||) instead. The factor 8 allows for
    # scalar conversion, multiply, add and float64 reduction rounding.
    eps = max(torch.finfo(t.dtype).eps for m,a in zip(main,target)
              if m is not None and a is not None for t in (m,a))
    scale = math.sqrt(before['main_sq'])*math.sqrt(before['aux_sq']) + abs(coefficient)*before['main_sq']
    tolerance = max(legacy_tolerance, 8*eps*scale)
    if not math.isfinite(tolerance) or after['dot'] < -tolerance:
        raise FloatingPointError(
            'projection failed to remove negative dot: before_dot=%r after_dot=%r '
            'tolerance=%r dtype_epsilon=%r main_sq=%r aux_sq=%r coefficient=%r' %
            (before['dot'], after['dot'], tolerance, eps, before['main_sq'],
             before['aux_sq'], coefficient))
    warnings.warn('Gradient projection residual is within dtype-scaled roundoff tolerance; '
                  'the original correction is unchanged.', RuntimeWarning)


def correction_for(main, target):
    before = geometry(main, target)
    coefficient = (-before['dot']/before['main_sq']
                   if before['dot'] < 0 and before['main_sq'] > 1e-24 else 0.)
    correction = tuple(m.detach()*coefficient if coefficient and m is not None and a is not None
                       else None for m,a in zip(main,target))
    after = geometry(main, add_gradients(target, correction)) if coefficient else before
    check_projection(before, after, main, target, coefficient)
    return correction, before, after, coefficient


class GradientGuard:
    def __init__(self, mode):
        if mode not in MODES[1:]:
            raise ValueError('guard requires observe, kd or all')
        self.mode = mode
        self.rows = {}

    def prepare(self, main_loss, terms, parameters, epoch):
        """Inspect the SAME forward graphs; .grad and global random states untouched."""
        gt, kd, modality = terms
        gradients = []
        for loss in (main_loss, gt, kd):
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite gradient-guard loss')
            gradients.append(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))
        main, g_gt, g_kd = gradients
        total = add_gradients(g_gt, g_kd)
        target = g_kd if self.mode == 'kd' else total
        if self.mode == 'observe':
            correction = tuple(None for _ in parameters)
            before = geometry(main, target)
            after, coefficient = before, 0.
        else:
            correction, before, after, coefficient = correction_for(main, target)
        details = dict(gt=geometry(main,g_gt), kd=geometry(main,g_kd), total=geometry(main,total))
        key = (epoch+1, modality)
        row = self.rows.setdefault(key, dict(epoch=epoch+1, modality=modality, batches=0,
                    projections=0, shared_tensors=before['shared'], correction_norm_sum=0.,
                    target_cosine_after_sum=0., total_cosine_after_sum=0., zero_main_batches=0,
                    **{name+suffix:0. for name in details for suffix in
                       ('_cosine_sum','_norm_sum','_main_norm_sum','_conflicts')}))
        row['batches'] += 1
        row['projections'] += int(coefficient > 0)
        row['zero_main_batches'] += int(before['main_sq'] <= 1e-24)
        row['correction_norm_sum'] += coefficient*math.sqrt(before['main_sq'])
        row['target_cosine_after_sum'] += after['cosine']
        row['total_cosine_after_sum'] += geometry(main,add_gradients(total,correction))['cosine']
        for name, values in details.items():
            row[name+'_cosine_sum'] += values['cosine']
            row[name+'_norm_sum'] += math.sqrt(values['aux_sq'])
            row[name+'_main_norm_sum'] += math.sqrt(values['main_sq'])
            row[name+'_conflicts'] += int(values['dot'] < 0)
        return correction

    @staticmethod
    def apply(parameters, correction):
        # Ordinary combined-loss backward runs first, preserving its exact rounding
        # in observe mode and on nonconflicting batches.
        with torch.no_grad():
            for parameter, delta in zip(parameters, correction):
                if delta is not None:
                    if parameter.grad is None:
                        raise RuntimeError('shared parameter missing combined gradient')
                    parameter.grad.add_(delta)
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('nonfinite gradient before optimizer step')

    def epoch_report(self, epoch):
        rows = [r for (e,_),r in self.rows.items() if e == epoch+1]
        count = sum(r['batches'] for r in rows)
        return dict(batches=count, projections=sum(r['projections'] for r in rows),
                    gt_conflicts=sum(r['gt_conflicts'] for r in rows),
                    kd_conflicts=sum(r['kd_conflicts'] for r in rows),
                    total_conflicts=sum(r['total_conflicts'] for r in rows))

    def summary(self):
        rows = [dict(self.rows[k]) for k in sorted(self.rows)]
        return dict(recipe=recipe(self.mode), rows=rows,
                    batches=sum(r['batches'] for r in rows),
                    projections=sum(r['projections'] for r in rows))
