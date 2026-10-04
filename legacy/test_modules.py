"""Numerical contract tests; portable legacy oracles plus local source parity.

The independent E/U oracle below is frozen from v37 qmd_complement.py,
qmd_supp.py and retained_u.py (unbounded_s).  It does not call new module
implementation helpers. When the original v37 package is available, its actual
AST classes/method are also executed directly without timm/trainer imports.
"""
import ast
import copy
import io
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from modules import OrderedHead, EBranch, UBranch, centered_loss, e_objective, u_objective


class LegacyE(nn.Module):
    """Frozen MLP_GT equations/initialization, with original packed-free API."""
    def __init__(self, dim, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 20260915)
            self.net = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.net[-1].weight)
        self.register_buffer('feature_mean', torch.zeros(dim))
        self.register_buffer('feature_scale', torch.ones(dim))
        self.register_buffer('global_bias', torch.zeros(()))
        self.register_buffer('calibrated', torch.tensor(False))

    def fit_training_statistics(self, cache):
        base, feat, y = [v.detach() for v in cache]
        self.feature_mean.copy_(feat.double().mean(0))
        self.feature_scale.copy_(feat.double().std(0, unbiased=False).clamp_min(.01))
        self.global_bias.copy_((y.double().reshape(-1) - base.double().reshape(-1)).mean())
        self.calibrated.fill_(True)

    def forward(self, features):
        x = ((features.detach() - self.feature_mean) / self.feature_scale).clamp(-5., 5.)
        return .03 * torch.tanh(self.net(x).reshape(-1))

    def evaluation_score(self, base, feature):
        return base.detach().reshape(-1).double() + self.global_bias.double() + self(feature).double()


class LegacyU(nn.Module):
    """Frozen unbounded_s equations, including legacy [fused|modal] input."""
    def __init__(self, dim, seed):
        super().__init__()
        self.dim = dim
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 20260916)
            heads = []
            for _ in range(3):
                head = nn.Sequential(nn.Linear(dim // 3, 16), nn.GELU(), nn.Linear(16, 1, bias=False))
                nn.init.zeros_(head[-1].weight)
                heads.append(head)
            self.heads = nn.ModuleList(heads)
        self.register_buffer('feature_mean', torch.zeros(dim))
        self.register_buffer('feature_scale', torch.ones(dim))
        self.register_buffer('bound', torch.zeros(()))
        self.register_buffer('calibrated', torch.tensor(False))
        self.register_buffer('gate', torch.ones(()))

    def fit_training_statistics(self, scores, packed):
        x = packed.detach()[:, self.dim:]
        self.feature_mean.copy_(x.double().mean(0))
        self.feature_scale.copy_(x.double().std(0, unbiased=False).clamp_min(.01))
        spread = float(scores.detach().double().std(unbiased=False))
        self.bound.fill_(min(.003, .1 * spread))
        self.calibrated.fill_(True)

    def components(self, packed):
        x = ((packed.detach()[:, self.dim:] - self.feature_mean) / self.feature_scale).clamp(-5., 5.)
        raw = self.bound * torch.cat([head(v) for head, v in zip(self.heads, x.chunk(3, dim=1))], 1)
        return self.gate.clamp(0., 1.) * raw

    def forward(self, packed):
        return self.components(packed).mean(1)

    def optimizer(self):
        return torch.optim.AdamW([dict(params=list(self.heads.parameters()), lr=.0001, weight_decay=.01)])


def legacy_centered(corrected, labels, residual):
    # Literal retained centered_supervision formula, no new helper calls.
    centered = F.mse_loss(corrected - corrected.mean(), labels.reshape(-1) - labels.mean())
    zero_mean = residual.mean().square()
    return centered + .1 * zero_mean


def independent_ordered(head, features, labels):
    """Equivalent soft labels constructed by indexing, not candidate logic."""
    embedding = F.normalize(head.quality_projector(features), dim=1)
    prototypes = F.normalize(head.quality_prototypes, dim=1)
    classes = torch.bucketize(labels.detach(), head.ordinal_thresholds)
    targets = torch.zeros(features.shape[0], 4, dtype=features.dtype, device=features.device)
    for index, category in enumerate(classes.tolist()):
        targets[index, category] = 1.
        if category > 0:
            targets[index, category - 1] = .1
        if category < 3:
            targets[index, category + 1] = .1
    targets /= targets.sum(1, keepdim=True)
    log_prob = F.log_softmax(embedding @ prototypes.t() / .1, dim=1)
    fitting = -(targets * log_prob).sum(1).mean()
    location = prototypes @ F.normalize(head.quality_axis, dim=0)
    ordering = F.relu(.1 - (location[1:] - location[:-1])).mean()
    return fitting + .1 * ordering


def legacy_source_root():
    parent = Path(__file__).resolve().parent.parent
    for name in ('v17_fs1000_v37', 'PAMFN-v17-fs1000-v37'):
        root = parent / name
        if (root / 'trainer/models/pamfn.py').is_file():
            return root
    return None


def extract_definition(path, name, environment, class_name=None):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    fragment = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(fragment), str(path), 'exec'), environment)
    return environment[name]


class ModuleContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(734)
        self.x = torch.randn(12, 8)
        self.modal = torch.randn(12, 12)
        self.base = torch.rand(12)
        self.labels = torch.rand(12)

    def assert_state_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(a, b)

    def assert_params_equal(self, a, b, grads=False):
        self.assertEqual(len(list(a.parameters())), len(list(b.parameters())))
        for left, right in zip(a.parameters(), b.parameters()):
            if grads:
                self.assert_state_equal(left.grad, right.grad)
            else:
                self.assert_state_equal(left, right)

    def test_private_rng_and_reproducible_initialization(self):
        for constructor in (lambda s: OrderedHead(8, [.2, .5, .8], s),
                            lambda s: EBranch(8, s), lambda s: UBranch(12, s)):
            state = torch.random.get_rng_state().clone()
            first = constructor(3)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            second = constructor(3)
            self.assert_state_equal(first.state_dict(), second.state_dict())
            third = constructor(4)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters())))
        if torch.cuda.is_available():
            before = torch.cuda.get_rng_state_all()
            OrderedHead(8, [.2, .5, .8], 9); EBranch(8, 9); UBranch(12, 9)
            self.assert_state_equal(before, torch.cuda.get_rng_state_all())

    def test_ordered_formula_and_gradients(self):
        head = OrderedHead(8, [.2, .5, .8], 2)
        other = copy.deepcopy(head)
        x = self.x.clone().requires_grad_(True)
        y = self.x.clone().requires_grad_(True)
        # Covers exact threshold boundaries and outside [0, 1].
        labels = torch.tensor([-.1, .0, .2, .21, .3, .5, .51, .7, .8, .81, 1., 1.1])
        actual = head.loss(x, labels)
        expected = independent_ordered(other, y, labels)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward(); expected.backward()
        self.assert_params_equal(head, other, grads=True)
        self.assertTrue(torch.equal(x.grad, y.grad))
        self.assertGreater(float(x.grad.abs().sum()), 0.)

    def test_ordered_zero_weight_preserves_main_optimizer_multistep(self):
        plain = nn.Sequential(nn.Linear(8, 8), nn.Dropout(.3), nn.Linear(8, 1))
        augmented = copy.deepcopy(plain)
        head = OrderedHead(8, [.2, .5, .8], 1)
        left = torch.optim.Adam(plain.parameters(), lr=.001)
        right = torch.optim.Adam(list(augmented.parameters()) + list(head.parameters()), lr=.001)
        for _ in range(4):
            state = torch.random.get_rng_state().clone()
            left.zero_grad()
            raw = plain(self.x).reshape(-1)
            F.mse_loss(raw, self.labels).backward(); left.step()
            after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(state)
            right.zero_grad()
            feature = augmented[0](self.x)
            predicted = augmented[2](augmented[1](feature)).reshape(-1)
            (F.mse_loss(predicted, self.labels) + 0. * head.loss(feature, self.labels)).backward()
            right.step()
            self.assertTrue(torch.equal(after, torch.random.get_rng_state()))
            self.assert_params_equal(plain, augmented)
            for a, b in zip(plain.parameters(), augmented.parameters()):
                self.assert_state_equal(left.state[a], right.state[b])

    def compare_e(self, reference):
        actual = EBranch(8, 2)
        actual.fit(self.base, self.x, self.labels)
        reference.fit_training_statistics((self.base, self.x, self.labels))
        self.assert_state_equal(actual.state_dict(), reference.state_dict())
        opt_a = actual.optimizer()
        opt_b = torch.optim.AdamW(reference.parameters(), lr=.0001, weight_decay=.01)
        for step in range(4):
            x = self.x.roll(step, 0)
            opt_a.zero_grad(); opt_b.zero_grad()
            left, _ = e_objective(actual, self.base, x, self.labels)
            residual = reference(x)
            right = legacy_centered(self.base + reference.global_bias + residual, self.labels, residual)
            self.assertEqual(left.dtype, torch.float32)
            self.assertTrue(torch.equal(left, right))
            left.backward(); right.backward()
            self.assert_params_equal(actual, reference, grads=True)
            opt_a.step(); opt_b.step()
            self.assert_state_equal(actual.state_dict(), reference.state_dict())
            self.assert_state_equal(opt_a.state_dict(), opt_b.state_dict())
            self.assertTrue(torch.equal(actual.evaluation_score(self.base, x), reference.evaluation_score(self.base, x)))

    def test_e_legacy_equations_gradients_and_adamw(self):
        self.compare_e(LegacyE(8, 2))

    def compare_u(self, reference):
        actual = UBranch(12, 2)
        frozen = self.base.double() + .17
        packed = torch.cat((torch.zeros_like(self.modal), self.modal), 1)
        actual.fit(frozen, self.modal)
        reference.fit_training_statistics(frozen, packed)
        for key, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[key]))
        opt_a, opt_b = actual.optimizer(), reference.optimizer()
        for step in range(4):
            x, old_x = self.modal.roll(step, 0), packed.roll(step, 0)
            opt_a.zero_grad(); opt_b.zero_grad()
            left, _ = u_objective(actual, frozen, x, self.labels)
            residual = reference(old_x).double()
            right = legacy_centered(frozen + residual, self.labels.double(), residual)
            self.assertEqual(left.dtype, torch.float64)
            self.assertTrue(torch.equal(left, right))
            left.backward(); right.backward()
            self.assert_params_equal(actual, reference, grads=True)
            opt_a.step(); opt_b.step()
            self.assert_params_equal(actual, reference)
            self.assert_state_equal(opt_a.state_dict(), opt_b.state_dict())
            self.assertTrue(torch.equal(actual.components(x), reference.components(old_x)))

    def test_u_legacy_equations_gradients_and_adamw(self):
        self.compare_u(LegacyU(12, 2))

    def test_no_gradient_to_e_u_parent(self):
        base = self.base.clone().requires_grad_(True)
        features = self.x.clone().requires_grad_(True)
        modal = self.modal.clone().requires_grad_(True)
        labels = self.labels.clone().requires_grad_(True)
        e, u = EBranch(8), UBranch(12)
        e.fit(base, features, labels)
        e_objective(e, base, features, labels)[0].backward()
        self.assertIsNone(base.grad); self.assertIsNone(features.grad); self.assertIsNone(labels.grad)
        e.zero_grad(set_to_none=True)
        frozen_score = e.evaluation_score(base, features)
        self.assertTrue(frozen_score.requires_grad)
        u.fit(frozen_score, modal)
        u_objective(u, frozen_score, modal, labels)[0].backward()
        self.assertIsNone(base.grad); self.assertIsNone(features.grad); self.assertIsNone(modal.grad)
        self.assertTrue(all(p.grad is None for p in e.parameters()))
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in u.parameters()))

    def test_calibration_immutable_and_training_only(self):
        e, u = EBranch(8), UBranch(12)
        e.fit(self.base, self.x, self.labels)
        u.fit(e.evaluation_score(self.base, self.x), self.modal)
        before = [copy.deepcopy(m.state_dict()) for m in (e, u)]
        with self.assertRaises(RuntimeError): e.fit(self.base, self.x + 99., self.labels)
        with self.assertRaises(RuntimeError): u.fit(self.base, self.modal + 99.)
        with torch.no_grad():
            e.evaluation_score(self.base, self.x + 99.)
            u(self.modal + 99.)
        for saved, model in zip(before, (e, u)):
            self.assert_state_equal(saved, model.state_dict())

    def test_invalid_shapes_nonfinite_unfitted(self):
        with self.assertRaises(ValueError): OrderedHead(8, [.2, .1, .8])
        with self.assertRaises(ValueError): OrderedHead(8, [float('nan'), .5, .8])
        with self.assertRaises(ValueError): UBranch(10)
        e, u = EBranch(8), UBranch(12)
        with self.assertRaises(RuntimeError): e(self.x)
        with self.assertRaises(RuntimeError): u(self.modal)
        bad = self.x.clone(); bad[0, 0] = float('nan')
        with self.assertRaises(ValueError): e.fit(self.base, bad, self.labels)
        with self.assertRaises(ValueError): e.fit(self.base, self.x[:, :7], self.labels)
        with self.assertRaises(ValueError): e.fit(self.base[:1], self.x[:1], self.labels[:1])
        with self.assertRaises(ValueError): e.fit(self.base.reshape(3, 4), self.x, self.labels)
        with self.assertRaises(ValueError): u.fit(self.base, self.modal[:, :9])
        with self.assertRaises(ValueError): u.fit(self.base * float('inf'), self.modal)
        self.assertFalse(bool(e.calibrated)); self.assertFalse(bool(u.calibrated))
        e.fit(self.base, self.x, self.labels); u.fit(self.base, self.modal)
        with self.assertRaises(ValueError): e(bad)
        with self.assertRaises(ValueError): u(self.modal * float('nan'))
        with self.assertRaises(ValueError): centered_loss(self.base, self.labels[:1], self.base)

    def test_zero_output_bias_only_preserves_tiny_gaps_double(self):
        e = EBranch(8)
        e.fit(self.base, self.x, self.labels)
        base = torch.tensor([.1, .10000001], dtype=torch.float32)
        features = self.x[:2]
        e.global_bias.fill_(100.)
        self.assertTrue(torch.equal(e(features), torch.zeros(2)))
        self.assertEqual(float(e.score(base, features)[0].detach()), float(e.score(base, features)[1].detach()))
        scored = e.evaluation_score(base, features)
        self.assertEqual(scored.dtype, torch.float64)
        self.assertLess(float(scored[0].detach()), float(scored[1].detach()))
        u = UBranch(12); u.fit(self.base, self.modal)
        self.assertTrue(torch.equal(u(self.modal), torch.zeros(12)))

    def test_u_fixed_mean_modality_order_and_not_hard_bounded(self):
        u = UBranch(12); u.fit(self.base, self.modal)
        with torch.no_grad():
            for index, head in enumerate(u.heads):
                head[0].weight.zero_(); head[0].bias.fill_(float(index + 1))
                head[-1].weight.fill_(100.)
        components = u.components(self.modal)
        self.assertTrue(bool((components[:, 0] < components[:, 1]).all()))
        self.assertTrue(bool((components[:, 1] < components[:, 2]).all()))
        self.assertTrue(bool((components.abs() > u.bound).all()))
        self.assertTrue(torch.equal(u(self.modal), components.mean(1)))
        # Gate is deliberately not learnable: only three MLP head parameter sets.
        self.assertEqual(len(list(u.parameters())), 9)

    def test_u_relative_scale_including_constant_e(self):
        for scores in (torch.linspace(.5, .5001, 12).double(), torch.ones(12).double(), self.base.double()):
            u = UBranch(12); u.fit(scores, self.modal)
            expected = torch.tensor(min(.003, .1 * float(scores.std(unbiased=False))), dtype=torch.float32)
            self.assertTrue(torch.equal(u.bound, expected))

    def test_u_constant_bias_cannot_improve_srcc(self):
        u = UBranch(12); u.fit(self.base, self.modal)
        with torch.no_grad():
            for head in u.heads:
                head[0].weight.zero_(); head[0].bias.fill_(1.)
                head[-1].weight.fill_(1.)
            correction = u(self.modal).double()
            self.assertTrue(torch.equal(correction, correction[0].expand_as(correction)))
            self.assertNotEqual(float(correction[0]), 0.)
            # These scores and labels are all distinct; double additions keep
            # their ranks, so rank-Pearson is exactly the same Spearman value.
            base = torch.tensor([.5, .0, .75, .25, .125, .625, 1., .375, .875, 1.25, 1.125, 1.375], dtype=torch.float64)
            labels = base.roll(3)
            def rho(scores):
                ranks = torch.argsort(torch.argsort(scores)).double()
                target = torch.argsort(torch.argsort(labels)).double()
                ranks = ranks - ranks.mean(); target = target - target.mean()
                return (ranks * target).sum() / (ranks.square().sum() * target.square().sum()).sqrt()
            self.assertTrue(torch.equal(rho(base), rho(base + correction)))
            self.assertFalse(torch.equal(F.mse_loss(base, labels), F.mse_loss(base + correction, labels)))

    def test_checkpoint_reload_exact(self):
        e, u = EBranch(8, 2), UBranch(12, 2)
        e.fit(self.base, self.x, self.labels)
        opt = e.optimizer(); opt.zero_grad(); e_objective(e, self.base, self.x, self.labels)[0].backward(); opt.step()
        frozen = e.evaluation_score(self.base, self.x)
        u.fit(frozen, self.modal)
        opt_u = u.optimizer(); opt_u.zero_grad(); u_objective(u, frozen, self.modal, self.labels)[0].backward(); opt_u.step()
        expected = e.evaluation_score(self.base, self.x) + u(self.modal).double()
        buffer = io.BytesIO(); torch.save(dict(e=e.state_dict(), u=u.state_dict()), buffer); buffer.seek(0)
        state = torch.load(buffer, map_location='cpu')
        new_e, new_u = EBranch(8, 99), UBranch(12, 99)
        new_e.load_state_dict(state['e'], strict=True); new_u.load_state_dict(state['u'], strict=True)
        self.assertTrue(torch.equal(expected, new_e.evaluation_score(self.base, self.x) + new_u(self.modal).double()))
        with self.assertRaises(RuntimeError): new_e.fit(self.base, self.x, self.labels)

    def test_production_dimensions_and_counts(self):
        self.assertEqual(sum(p.numel() for p in EBranch().parameters()), 4128)
        self.assertEqual(sum(p.numel() for p in UBranch().parameters()), 12384)
        self.assertEqual(sum(p.numel() for p in OrderedHead(thresholds=[.2, .5, .8]).parameters()), 33536)

    @unittest.skipIf(legacy_source_root() is None, 'optional original v37 source not installed; frozen portable oracles remain active')
    def test_actual_archived_legacy_ast_multistep_parity(self):
        root = legacy_source_root()
        env = dict(torch=torch, nn=nn, F=F,
                   branch_config=lambda tag, input_dim: dict(input_dim=input_dim, hidden=16, bound=.03,
                                                            feature_clip=5., std_floor=.01))
        old_e = extract_definition(root / 'trainer/tools/qmd_complement.py', 'ComplementBranch', env)
        self.compare_e(old_e('ordered_bc_mlp16_gt', 8, 2))
        env.update(SCALED='e_modal_scaled', LR=.0001, GATE_LR=.001, GATE_PENALTY=1.,
                   RECIPES={'e_modal_scaled': dict(shrink=False, relative=True)},
                   branch_config=lambda tag, input_dim: dict(input_dim=input_dim, recipe=dict(relative=True, shrink=False),
                                                            abs_bound=.003, relative_ratio=.1))
        extract_definition(root / 'trainer/tools/qmd_supp.py', 'SupplementBranch', env)
        old_u = extract_definition(root / 'retained_u.py', 'ControlledSupplement', env)
        self.compare_u(old_u('unbounded_s', 12, 2))
        old_loss = extract_definition(root / 'trainer/models/pamfn.py', '_ordinal_prototype_loss',
                                     dict(torch=torch, F=F), class_name='PAMFN')
        head = OrderedHead(8, [.2, .5, .8], 4)
        expected = old_loss(head, F.normalize(head.quality_projector(self.x), dim=1), self.labels)
        actual = head.loss(self.x, self.labels)
        self.assertTrue(torch.equal(actual, expected))
        left = torch.autograd.grad(actual, tuple(head.parameters()))
        right = torch.autograd.grad(expected, tuple(head.parameters()))
        self.assert_state_equal(left, right)


if __name__ == '__main__':
    unittest.main(verbosity=2)
