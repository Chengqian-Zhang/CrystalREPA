# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional

from mattergen.diffusion.losses import SummedFieldLoss, denoising_score_matching
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import FieldLoss, d3pm_loss
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss


class MaterialsLoss(SummedFieldLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        weights: Optional[Dict[str, float]] = None,
        alignment_loss: str = "ea_nce",
        teacher_models: list | None = None,
        lambda_repa: float | None = None,
        element_aware: bool = True,
        tau: float = 0.1,
    ):
        if alignment_loss not in {"ea_nce", "info_nce", "negative_cosine"}:
            raise ValueError(f"Unknown alignment objective: {alignment_loss}")
        if alignment_loss == "ea_nce" and not element_aware:
            raise ValueError("EA-NCE requires element_aware=True")
        if alignment_loss == "info_nce" and element_aware:
            raise ValueError("Unmasked symmetric InfoNCE requires element_aware=False")
        self.element_aware = element_aware
        self.alignment_loss = alignment_loss
        self.tau = tau
        self.teacher_models = list(teacher_models) if teacher_models is not None else ["dpa_3_1_3m"]

        model_targets = {"pos": ModelTarget.score_times_std, "cell": ModelTarget.score_times_std}
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss,
                reduce=reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")
            loss_fns["atomic_numbers"] = partial(
                d3pm_loss,
                reduce=reduce,
                d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            )
        self.reduce = reduce
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            lambda_repa=lambda_repa,
            model_targets=model_targets,
        )
