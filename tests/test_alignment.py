"""Check EA-NCE against the paper's explicit masked log-softmax objective."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mattergen.common.loss import MaterialsLoss


class Batch:
    def __init__(self, elements, teacher):
        self.atomic_numbers = elements
        self.dpa_3_1_3m_rep = teacher
        self.num_nodes = len(elements)

    def get_batch_idx(self, field):
        return torch.zeros(self.num_nodes, dtype=torch.long)

    def get_batch_size(self):
        return 1


@pytest.mark.parametrize("masked", [True, False])
def test_alignment_loss_and_gradient_match_equation(monkeypatch, masked):
    # Isolate alignment while retaining a nonzero generation-loss contribution.
    monkeypatch.setattr("mattergen.diffusion.losses.apply", lambda **kw: {"pos": torch.tensor([0.25])})
    z = torch.tensor([[1., 0.2], [0.3, 1.], [-0.5, 0.7]], requires_grad=True)
    teacher = torch.tensor([[0.9, 0.1], [0.1, 0.8], [-0.6, 0.4]])
    elements = torch.tensor([1, 1, 8])
    batch = Batch(elements, teacher)
    criterion = MaterialsLoss(include_cell=False, include_atomic_numbers=False,
                             weights={"pos": 2.}, lambda_repa=0.7, tau=0.3,
                             alignment_loss="ea_nce" if masked else "info_nce",
                             element_aware=masked)
    loss, metrics = criterion(multi_corruption=SimpleNamespace(corruptions={"pos": None}),
                              batch=batch, noisy_batch=batch,
                              score_model_output=SimpleNamespace(student_features=[z]), t=torch.tensor([0.5]))
    reference_z = z.detach().clone().requires_grad_(True)
    scores = F.normalize(reference_z, dim=-1) @ F.normalize(teacher, dim=-1).T / 0.3
    terms = []
    for i in range(len(elements)):
        candidates = [j for j in range(len(elements)) if not masked or i == j or elements[i] != elements[j]]
        terms.append(torch.logsumexp(scores[i, candidates], dim=0) - scores[i, i])
        terms.append(torch.logsumexp(scores[candidates, i], dim=0) - scores[i, i])
    expected = torch.stack(terms).mean()
    torch.testing.assert_close(metrics["repa"], expected)
    torch.testing.assert_close(loss, 0.5 + 0.7 * expected)
    loss.backward()
    (0.5 + 0.7 * expected).backward()
    torch.testing.assert_close(z.grad, reference_z.grad)


def test_ea_nce_requires_element_mask():
    with pytest.raises(ValueError, match="element_aware"):
        MaterialsLoss(alignment_loss="ea_nce", element_aware=False)


def test_baseline_does_not_enable_alignment():
    assert not MaterialsLoss(weights={"pos": .1, "cell": 1., "atomic_numbers": 1.}).repa
