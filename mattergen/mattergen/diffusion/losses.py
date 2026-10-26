# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional, Protocol, Tuple, TypeVar

import torch
import torch.nn.functional as F

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTargets
from mattergen.diffusion.training.field_loss import FieldLoss, denoising_score_matching

T = TypeVar("T", bound=BatchedData)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class Loss(Protocol[T]):
    """Loss function for training a score model on multi-field data."""

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    """model_targets tells us what this loss function trains the score model to predict.
    We need this information in order to convert the model output to a score during sampling.
    """
    model_targets: ModelTargets


class SummedFieldLoss(Loss[T]):
    """(Weighted) sum of different loss functions applied on each field."""

    def __init__(
        self,
        loss_fns: Dict[str, FieldLoss],
        model_targets: ModelTargets,
        weights: Optional[Dict[str, float]] = None,
        lambda_repa: float | None = None,
    ) -> None:
        self.model_targets = model_targets
        self.loss_fns = loss_fns
        weights = dict(weights) if weights is not None else {k: 1.0 for k in loss_fns}
        self.lambda_repa = lambda_repa
        self.repa = self.lambda_repa is not None
        assert set(weights) == set(loss_fns), "Generation loss weights must match generation fields"
        self.loss_weights = weights

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_idx = {k: batch.get_batch_idx(k) for k in self.loss_fns.keys()}
        node_is_unmasked = {k: node_is_unmasked for k in self.loss_fns.keys()}

        # Dict[str, torch.Tensor]
        # Keys are field names and values are loss per sample, with shape (batch_size,).
        loss_per_sample_per_field = apply(
            fns=self.loss_fns,
            corruption=multi_corruption.corruptions,
            x=batch,
            noisy_x=noisy_batch,
            score_model_output=score_model_output,
            batch_idx=batch_idx,
            broadcast=dict(t=t, batch_size=batch.get_batch_size(), batch=batch),
            node_is_unmasked=node_is_unmasked,
        )
        assert set([v.shape for v in loss_per_sample_per_field.values()]) == {
            (batch.get_batch_size(),)
        }, "All losses should have shape (batch_size,)."
        # Aggregate losses per field over samples.
        scalar_loss_per_field = {k: v.mean() for k, v in loss_per_sample_per_field.items()}

        # Dict[str, torch.Tensor], dictionary containing metrics to be logged,
        metrics_dict = scalar_loss_per_field
        # This is the loss that is used for backpropagation (after mean aggregation over samples).
        # Shape: (batch_size,)
        agg_loss = torch.stack(
            [self.loss_weights[k] * v for k, v in loss_per_sample_per_field.items()], dim=0
        ).sum(0)

        if self.repa:
            student_features = score_model_output.student_features
            if self.teacher_models[0] == "dpa_3_1_3m":
                teacher_features = [batch.dpa_3_1_3m_rep.float()]
            elif self.teacher_models[0] == "dpa_3_1_mptrj":
                teacher_features = [batch.dpa_3_1_mptrj_rep.float()]
            elif self.teacher_models[0] == "dpa_3_1_3m_ft":
                teacher_features = [batch.dpa_3_1_3m_ft_rep.float()]
            elif self.teacher_models[0] == "mace_mpa_0":
                teacher_features = [batch.mace_mpa_0_rep.float()]
            elif self.teacher_models[0] == "mace_mp_0":
                teacher_features = [batch.mace_mp_0_rep.float()]
            elif self.teacher_models[0] == "mattersim_v1_5m":
                teacher_features = [batch.mattersim_v1_5m_rep.float()]
            elif self.teacher_models[0] == "orb_v2":
                teacher_features = [batch.orb_v2_rep.float()]
            elif self.teacher_models[0] == "orb_v3":
                teacher_features = [batch.orb_v3_rep.float()]
            elif self.teacher_models[0] == "sevennet_l3i5":
                teacher_features = [batch.sevennet_l3i5_rep.float()]
            elif self.teacher_models[0] == "sevennet_omni_i12":
                teacher_features = [batch.sevennet_omni_i12_rep.float()]
            else:
                raise RuntimeError(f"Do not support {self.teacher_models[0]}.")

            assert len(teacher_features) == len(student_features)
            # Each experiment aligns to one frozen teacher.
            assert len(teacher_features) == 1
            z = student_features[0]
            z_tilde = teacher_features[0]

            repa_loss = 0.0
            if self.alignment_loss in {"ea_nce", "info_nce"}:
                # Paper Eq. (6)-(8): z is projected student, z_tilde is frozen teacher.
                # Cross entropy evaluates the symmetric EA-NCE log-softmax terms.
                z = F.normalize(z, dim=-1)
                z_tilde = F.normalize(z_tilde, dim=-1)
                # Compute similarity matrix
                similarity_logits = torch.matmul(z, z_tilde.T) / self.tau
                curr_device = similarity_logits.device
                if self.element_aware:
                    diff_type_mask = (batch.atomic_numbers.unsqueeze(0) != batch.atomic_numbers.unsqueeze(1)) # [N, N]
                    eye_mask = torch.eye(batch.num_nodes, device=curr_device).bool()
                    element_mask = diff_type_mask | eye_mask  # Paper mask M, Eq. (7).
                    similarity_logits = similarity_logits.masked_fill(~element_mask, -9e15)
                # Generate labels
                labels = torch.arange(batch.num_nodes, device=curr_device)
                # Compute loss
                loss_student_to_teacher = F.cross_entropy(similarity_logits, labels)
                loss_teacher_to_student = F.cross_entropy(similarity_logits.T, labels)
                repa_loss += (loss_student_to_teacher + loss_teacher_to_student) / 2
            elif self.alignment_loss == "negative_cosine":
                z = torch.nn.functional.normalize(z, dim=-1)
                z_tilde = torch.nn.functional.normalize(z_tilde, dim=-1)
                repa_loss += mean_flat(-(z_tilde * z).sum(dim=-1))
            else:
                raise RuntimeError(f"Unsupported alignment objective: {self.alignment_loss}")

            metrics_dict["repa"] = repa_loss
            return (
                agg_loss.mean() + self.lambda_repa * repa_loss,
                metrics_dict,
            )
        else:
            return (
                agg_loss.mean(),
                metrics_dict,
            )


class DenoisingScoreMatchingLoss(SummedFieldLoss):
    def __init__(
        self,
        model_targets: ModelTargets,
        reduce: Literal["sum", "mean"] = "mean",
        weights: Optional[Dict[str, float]] = None,
        field_center_zero: Optional[Dict[str, bool]] = None,  # Whether to zero center each field.
    ):
        if field_center_zero is not None:
            assert set(field_center_zero.keys()) == set(model_targets.keys())

        super().__init__(
            loss_fns={
                k: partial(
                    denoising_score_matching,
                    reduce=reduce,
                    model_target=v,
                )
                for k, v in model_targets.items()
            },
            model_targets=model_targets,
            weights=weights,
        )
