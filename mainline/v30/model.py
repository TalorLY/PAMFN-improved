"""Reusable one-backbone inference and predeclared controls; v17 itself is unchanged."""
import torch
from contextlib import contextmanager
from torch import nn
from tools.qmd_modal import modal_view
from tools.qmd_supp import SupplementBranch, e_score
from tools.qmd_supp_config import SCALED
from tools.qmd_residual import assert_frozen
from tools.train_qmd_residual import metrics
from tools.qmd_pair import pair_metrics
from tools.qmd_complement import ComplementBranch
from tools.qmd_complement_config import MLP_GT


class ZeroE(nn.Module):
    def evaluation_score(self, base, packed): return base.double()


class RGBBranch(ComplementBranch):
    """Same E rule on the 256-D pooled feature of the existing RGB-only scorer."""
    def __init__(self, input_dim=256, seed=0):
        super().__init__(MLP_GT,input_dim,seed)
        self.config.update(heads='joint',features='rgb_only',transfer='pretrained single-modality scorer')


def rgb_view(anchor, inputs):
    assert_frozen(anchor)
    with torch.no_grad():
        score,info=anchor(inputs)
        feature=info['feats'][-1].squeeze(2).detach()
    return score.reshape(-1).detach(),feature


def raw_supervision(corrected, label, residual):
    """Change only batch centering; keep mean penalty, capacity and optimizer."""
    from torch.nn import functional as F
    raw=F.mse_loss(corrected,label.reshape(-1))
    mean=residual.mean().square()
    return raw+.1*mean,raw,mean


@contextmanager
def control_context(method):
    # Isolated worker process only. Restore original module functions even on failure.
    from tools import qmd_modal
    view,supervision=qmd_modal.modal_view,qmd_modal.centered_supervision
    try:
        if method=='rgb_e': qmd_modal.modal_view=rgb_view
        if method=='raw_e': qmd_modal.centered_supervision=raw_supervision
        yield
    finally:
        qmd_modal.modal_view=view;qmd_modal.centered_supervision=supervision


class ControlledSupplement(SupplementBranch):
    """Keep dimensions/initial weights/optimizer/budget; change one stated property."""
    def __init__(self, control, input_dim=768, seed=0):
        if control not in ('fused_s', 'unbounded_s'): raise ValueError('unknown control')
        super().__init__(SCALED, input_dim, seed)
        self.control = control
        self.config = dict(self.config, paper_control=control)

    def feature_input(self, packed):
        if self.control == 'fused_s':
            dim = self.config['input_dim']
            return torch.cat((packed[:, :dim], packed[:, :dim]), dim=1)
        return packed

    def fit_training_statistics(self, e_scores, packed):
        stats = super().fit_training_statistics(e_scores, self.feature_input(packed))
        stats.update(feature_source='fused_partitioned' if self.control == 'fused_s' else 'modal', paper_control=self.control)
        return stats

    def raw_components(self, packed):
        packed = self.feature_input(packed)
        if self.control != 'unbounded_s': return super().raw_components(packed)
        if not bool(self.calibrated): raise RuntimeError('fit training statistics first')
        dim = self.config['input_dim']
        x = ((packed.detach()[:, dim:] - self.feature_mean) / self.feature_scale).clamp(-5., 5.)
        # Same local derivative at zero as bound*tanh(z); only tanh is removed.
        return self.bound * torch.cat([h(v) for h, v in zip(self.heads, x.chunk(3, dim=1))], 1)


def predict_views(anchor, e, supplement, inputs):
    """Exactly one PAMFN forward. Preserve v17 float32 mean then float64 additions."""
    assert_frozen(anchor); assert_frozen(e)
    with torch.no_grad():
        base, packed = modal_view(anchor, inputs)
        corrected = e_score(e, base, packed)
        return dict(ordered=base.double(), e=corrected, v17=corrected + supplement(packed).double())


def diagnose(e, supplement, cache):
    """Post-hoc interventions, NOT independently trained ablations or chosen variants."""
    with torch.no_grad():
        frozen = e_score(e, cache[0], cache[1]); raw = supplement.components(cache[1])
        out = {}
        for name, scale, missing in [('full', 1., None), ('zero', 0., None), ('half', .5, None), ('double', 2., None),
                                     ('without_V', 1., 0), ('without_F', 1., 1), ('without_A', 1., 2)]:
            components = raw.clone()
            if missing is not None: components[:, missing] = 0
            # Keep denominator 3; removing a modality must not amplify the others.
            delta = components.mean(1).double() * scale
            out[name] = dict(metrics(frozen + delta, cache[2]), **pair_metrics(frozen, delta, cache[2]))
        return dict(kind='fixed-checkpoint diagnostic; no test-selected setting', denominator=3, variants=out)
