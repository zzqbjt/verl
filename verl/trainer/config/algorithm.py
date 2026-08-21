# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = [
    "AlgoConfig",
    "DapoReferenceKLConfig",
    "FilterGroupsConfig",
    "KLControlConfig",
    "RolloutCorrectionConfig",
    "StepSplitConfig",
    "StepValueAdvantageConfig",
]


@dataclass
class DapoReferenceKLConfig(BaseConfig):
    """Auxiliary privileged-context KL loss for DAPO.

    For every rollout group retained by DAPO, a verified-correct sampled response
    is inserted into the prompt used by the rollout old policy.  Its compressed
    teacher distribution is cached before any actor optimizer step.  The old
    teacher and the current ordinary-context actor score the same incorrect
    response; their forward KL is mixed with the normal DAPO policy loss.
    """

    enabled: bool = False
    loss_coef: float = 0.0
    teacher_prompt_template: str = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n\n{question}\n\nRemember to put "
        'your answer on its own line after "Answer:".\n\nHere is a reference solution: '
        "{reference_solution}"
    )
    question_key: str = "question"
    student_prompt_prefix: str = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n\n"
    )
    student_prompt_suffix: str = '\n\nRemember to put your answer on its own line after "Answer:".'
    max_teacher_prompt_length: int = 2048
    truncation: str = "error"
    temperature: float = 1.0
    approximation: str = "topk"
    top_k: int = 100
    token_chunk_size: int = 512
    teacher_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"dapo_reference_kl.enabled must be a bool, got {self.enabled!r}.")
        if (
            not isinstance(self.loss_coef, (int, float))
            or isinstance(self.loss_coef, bool)
            or not math.isfinite(self.loss_coef)
            or not 0.0 <= self.loss_coef <= 1.0
        ):
            raise ValueError("dapo_reference_kl.loss_coef must be finite and in [0, 1].")
        if not isinstance(self.teacher_prompt_template, str) or not self.teacher_prompt_template.strip():
            raise ValueError("dapo_reference_kl.teacher_prompt_template must be a non-empty string.")
        for placeholder in ("{question}", "{reference_solution}"):
            if placeholder not in self.teacher_prompt_template:
                raise ValueError(f"dapo_reference_kl.teacher_prompt_template must contain {placeholder!r}.")
        if not isinstance(self.question_key, str) or not self.question_key.strip():
            raise ValueError("dapo_reference_kl.question_key must be a non-empty string.")
        if not isinstance(self.student_prompt_prefix, str):
            raise ValueError("dapo_reference_kl.student_prompt_prefix must be a string.")
        if not isinstance(self.student_prompt_suffix, str):
            raise ValueError("dapo_reference_kl.student_prompt_suffix must be a string.")
        if (
            not isinstance(self.max_teacher_prompt_length, int)
            or isinstance(self.max_teacher_prompt_length, bool)
            or self.max_teacher_prompt_length < 1
        ):
            raise ValueError("dapo_reference_kl.max_teacher_prompt_length must be an integer >= 1.")
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError("dapo_reference_kl.truncation must be one of: error, left, right.")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("dapo_reference_kl.temperature must be finite and > 0.")
        if self.approximation != "topk":
            raise ValueError("Old-policy dapo_reference_kl.approximation must be 'topk'.")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("dapo_reference_kl.top_k must be an integer >= 1.")
        if (
            not isinstance(self.token_chunk_size, int)
            or isinstance(self.token_chunk_size, bool)
            or self.token_chunk_size < 1
        ):
            raise ValueError("dapo_reference_kl.token_chunk_size must be an integer >= 1.")


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class StepSplitConfig(BaseConfig):
    """Configuration for tokenizer-aligned reasoning-step boundaries.

    Step splitting is a reusable batch annotation and does not imply that a
    value probe, a particular advantage estimator, or hidden-state extraction
    is enabled.

    Args:
        enabled (bool): Attach ``step_end_mask`` for methods that do not
            require it. V-Info and step-value estimators enable splitting
            automatically.
        lookahead_tokens (int): Tokens inspected after a delimiter for a numeric step marker.
        separate_preamble (bool): Whether text before an explicit Step 1 marker is a separate step.
    """

    enabled: bool = False
    lookahead_tokens: int = 10
    separate_preamble: bool = False

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"step_split.enabled must be a bool, got {self.enabled!r}.")
        if (
            not isinstance(self.lookahead_tokens, int)
            or isinstance(self.lookahead_tokens, bool)
            or self.lookahead_tokens < 1
        ):
            raise ValueError(f"step_split.lookahead_tokens must be an integer >= 1, got {self.lookahead_tokens!r}.")
        if not isinstance(self.separate_preamble, bool):
            raise ValueError(f"step_split.separate_preamble must be a bool, got {self.separate_preamble!r}.")


@dataclass
class StepValueAdvantageConfig(BaseConfig):
    """Configuration for constructing advantages from provider-supplied step values.

    The estimator consumes ``step_values`` but does not prescribe how they are
    produced. ``provider`` selects the built-in trainable probe or same-prompt
    semantic retrieval while the surrounding split and advantage stages stay shared.

    Args:
        provider (str): Step-value estimator used between shared splitting and shared advantage computation.
        lam (float): Decay applied to future step-value differences.
        norm_by_group_std (bool): Whether the provider prepares step-value scales from its target's prompt-group
            standard deviation. Probe targets use acc space; retrieval targets may use raw-reward space.
        zero_when_group_uniform (bool): Whether groups uniform in the provider's target space receive zero advantage.
        target_key (str): Reward-manager output key used as the probe target.
        task_reward_key (str): Reward-manager output key for the verifier's raw task reward.
        prompt_center_calibration_enabled (bool): Apply a frozen rank-preserving prompt-center map to Probe logits.
        prompt_center_calibration_slope (float): Positive slope of the frozen prompt-center map.
        prompt_center_calibration_intercept (float): Intercept of the frozen prompt-center map.
        prompt_center_audit_enabled (bool): Learn a causally lagged prompt-center map from uncensored DAPO groups.
        prompt_center_audit_groups (int): Complete first-generation prompt groups audited per update.
        prompt_center_audit_window (int): Number of completed audit updates retained by the online fit.
        prompt_center_audit_seed (int): Seed used only for outcome-independent first-batch ordinal priorities.
        similarity_top_k (int): Number of semantic neighbors used by the similarity provider.
        similarity_tau (float): Softmax temperature for similarity weights.
        similarity_position_window (float): Maximum relative-position distance between retrieved steps.
        similarity_iterations (int): Number of synchronous value-propagation rounds.
        lookahead_tokens (Optional[int]): Deprecated compatibility alias for ``step_split.lookahead_tokens``.
        separate_preamble (Optional[bool]): Deprecated compatibility alias for ``step_split.separate_preamble``.
    """

    provider: str = "probe"
    lam: float = 0.9
    norm_by_group_std: bool = True
    zero_when_group_uniform: bool = True
    target_key: str = "acc"
    task_reward_key: str = "score"
    prompt_center_calibration_enabled: bool = False
    prompt_center_calibration_slope: float = 1.0
    prompt_center_calibration_intercept: float = 0.0
    prompt_center_audit_enabled: bool = False
    prompt_center_audit_groups: int = 16
    prompt_center_audit_window: int = 2
    prompt_center_audit_seed: int = 0
    similarity_top_k: int = 3
    similarity_tau: float = 0.002
    similarity_position_window: float = 0.2
    similarity_iterations: int = 1
    lookahead_tokens: Optional[int] = None
    separate_preamble: Optional[bool] = None

    def __post_init__(self):
        """Validate step-value advantage parameters."""
        if self.provider not in {"probe", "similarity"}:
            raise ValueError(f"step_value.provider must be 'probe' or 'similarity', got {self.provider!r}.")
        if (
            not isinstance(self.lam, (int, float))
            or isinstance(self.lam, bool)
            or not math.isfinite(self.lam)
            or not 0.0 <= self.lam <= 1.0
        ):
            raise ValueError(f"step_value.lam must be finite and in [0, 1], got {self.lam}.")
        if not isinstance(self.norm_by_group_std, bool):
            raise ValueError(f"step_value.norm_by_group_std must be a bool, got {self.norm_by_group_std!r}.")
        if not isinstance(self.zero_when_group_uniform, bool):
            raise ValueError(
                f"step_value.zero_when_group_uniform must be a bool, got {self.zero_when_group_uniform!r}."
            )
        if not isinstance(self.target_key, str) or not self.target_key.strip():
            raise ValueError(f"step_value.target_key must be a non-empty string, got {self.target_key!r}.")
        if not isinstance(self.task_reward_key, str) or not self.task_reward_key.strip():
            raise ValueError(f"step_value.task_reward_key must be a non-empty string, got {self.task_reward_key!r}.")
        if not isinstance(self.prompt_center_calibration_enabled, bool):
            raise ValueError(
                "step_value.prompt_center_calibration_enabled must be a bool, "
                f"got {self.prompt_center_calibration_enabled!r}."
            )
        if (
            isinstance(self.prompt_center_calibration_slope, bool)
            or not isinstance(self.prompt_center_calibration_slope, (int, float))
            or not math.isfinite(self.prompt_center_calibration_slope)
            or self.prompt_center_calibration_slope <= 0.0
        ):
            raise ValueError(
                "step_value.prompt_center_calibration_slope must be finite and positive, "
                f"got {self.prompt_center_calibration_slope!r}."
            )
        if (
            isinstance(self.prompt_center_calibration_intercept, bool)
            or not isinstance(self.prompt_center_calibration_intercept, (int, float))
            or not math.isfinite(self.prompt_center_calibration_intercept)
        ):
            raise ValueError(
                "step_value.prompt_center_calibration_intercept must be finite, "
                f"got {self.prompt_center_calibration_intercept!r}."
            )
        if not isinstance(self.prompt_center_audit_enabled, bool):
            raise ValueError(
                "step_value.prompt_center_audit_enabled must be a bool, "
                f"got {self.prompt_center_audit_enabled!r}."
            )
        if self.prompt_center_audit_enabled and self.prompt_center_calibration_enabled:
            raise ValueError(
                "step_value.prompt_center_audit_enabled and prompt_center_calibration_enabled are mutually exclusive"
            )
        for name, value in (
            ("prompt_center_audit_groups", self.prompt_center_audit_groups),
            ("prompt_center_audit_window", self.prompt_center_audit_window),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"step_value.{name} must be a positive integer, got {value!r}.")
        if (
            isinstance(self.prompt_center_audit_seed, bool)
            or not isinstance(self.prompt_center_audit_seed, int)
            or self.prompt_center_audit_seed < 0
        ):
            raise ValueError(
                "step_value.prompt_center_audit_seed must be a non-negative integer, "
                f"got {self.prompt_center_audit_seed!r}."
            )
        if (
            isinstance(self.similarity_top_k, bool)
            or not isinstance(self.similarity_top_k, int)
            or self.similarity_top_k <= 0
        ):
            raise ValueError(f"step_value.similarity_top_k must be a positive integer, got {self.similarity_top_k!r}.")
        if (
            isinstance(self.similarity_tau, bool)
            or not isinstance(self.similarity_tau, (int, float))
            or not math.isfinite(self.similarity_tau)
            or self.similarity_tau <= 0
        ):
            raise ValueError(f"step_value.similarity_tau must be finite and positive, got {self.similarity_tau!r}.")
        if (
            not isinstance(self.similarity_position_window, (int, float))
            or not math.isfinite(self.similarity_position_window)
            or not 0.0 <= self.similarity_position_window <= 1.0
        ):
            raise ValueError(
                "step_value.similarity_position_window must be finite and in [0, 1], "
                f"got {self.similarity_position_window!r}."
            )
        if (
            isinstance(self.similarity_iterations, bool)
            or not isinstance(self.similarity_iterations, int)
            or self.similarity_iterations <= 0
        ):
            raise ValueError(
                f"step_value.similarity_iterations must be a positive integer, got {self.similarity_iterations!r}."
            )
        if self.lookahead_tokens is not None and (
            not isinstance(self.lookahead_tokens, int)
            or isinstance(self.lookahead_tokens, bool)
            or self.lookahead_tokens < 1
        ):
            raise ValueError(f"step_value.lookahead_tokens must be an integer >= 1, got {self.lookahead_tokens!r}.")
        if self.separate_preamble is not None and not isinstance(self.separate_preamble, bool):
            raise ValueError(f"step_value.separate_preamble must be a bool, got {self.separate_preamble!r}.")


@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Configuration for Rollout Correction (addresses off-policy issues in RL training).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Rollout Correction handles off-policiness from multiple sources:
    1. Policy mismatch: Rollout policy (e.g., vLLM BF16) vs Training policy (e.g., FSDP FP32)
    2. Model update staleness: Rollout data collected from older policy checkpoints
    3. General off-policy scenarios: Any distribution shift between data collection and training

    For more details, see:
    "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
    https://richardli.xyz/rl-collapse

    This typed config replaces the old dict-based approach and provides:
    - Type safety and validation
    - Clear documentation of all parameters
    - Named factory methods for common presets (TIS, MIS, etc.)
    - Sensible defaults

    Args:
        rollout_is (Optional[str]): IS weight aggregation level.
            - None: No IS weights (metrics only)
            - "token": Per-token IS weights (low variance, biased)
            - "sequence": Per-sequence IS weights (unbiased, high variance)
            Default: "sequence"

        rollout_is_threshold (float): Upper threshold for IS weight truncation/rejection.
            Typical range: 1.5-5.0 for token level, 2.0-10.0 for sequence level.
            Default: 2.0

        rollout_is_batch_normalize (bool): Apply batch normalization to IS weights.
            - True: Normalize IS weights to have mean=1.0 within each batch
            - False: Use raw (truncated) IS weights (standard)
            - Reduces variance by ensuring average weight is 1.0 per batch
            - Only affects IS weight values, not rejection sampling
            Default: False (no batch normalization)

        rollout_rs (Optional[str]): Rejection sampling aggregation modes.
            Accepts a comma-delimited list (duplicates removed) of canonical options implemented in
            ``rollout_corr_helper``:
            - "token_k1": Token-level rejection with ``-log r`` (ratio thresholds supplied via
              ``rollout_rs_threshold`` as ``lower_upper``)
            - "token_k2": Token-level rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "token_k3": Token-level rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            - "seq_sum_k1": Sequence sum of ``-log r`` (ratio bounds)
            - "seq_sum_k2": Sequence sum of rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "seq_sum_k3": Sequence sum of rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            - "seq_mean_k1": Sequence mean of ``-log r`` (ratio bounds)
            - "seq_mean_k2": Sequence mean of rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "seq_mean_k3": Sequence mean of rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            - "seq_max_k2": Sequence max of rejection with ``0.5 * (log r)^2`` (upper bound only)
            - "seq_max_k3": Sequence max of rejection with ``exp(log r) - 1 - log r`` (upper bound only)
            names automatically. Default: None

        rollout_rs_threshold (Optional[Union[str, float]]): Threshold specification for rejection sampling.
            Provide one value per option (single entry is broadcast when multiple options are supplied).
            Ratio-based modes (``*k1``) expect ``lower_upper`` strings; supplying a single float implies
            only the upper ratio bound, with the lower bound inferred as its reciprocal. Divergence modes
            (k2/k3) expect positive upper bounds (float or string). Default: None

        bypass_mode (bool): Operating mode - bypass or decoupled.
            - True: Bypass mode - reuse rollout_log_prob as old_log_prob (2 policies)
              Uses compute_policy_loss_bypass_mode() with loss_type selection
            - False: Decoupled mode - compute old_log_prob separately (3 policies)
              Uses standard PPO loss with IS weight correction
            Default: False (decoupled mode)

        loss_type (str): Loss function type in bypass mode (bypass_mode=True).
            - "reinforce": REINFORCE-style policy gradient with explicit IS weights
              L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
            - "ppo_clip": PPO clipped objective (IS handled by ratio, no explicit weights)
              L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Default: "ppo_clip"

    Example:
        # Create with defaults
        config = RolloutCorrectionConfig()

        # Decoupled PPO mode presets (3 policies: π_rollout, π_old, π_θ)
        # IS weights correct for gap between π_old and π_rollout
        config = RolloutCorrectionConfig.decoupled_token_is()  # Token-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is()    # Seq-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is_rs() # Seq-MIS
        config = RolloutCorrectionConfig.decoupled_geo_rs()    # Geo-RS (ratio mode)

        # Bypass mode presets (2 policies: π_rollout = π_old, π_θ)
        # loss_type controls the loss function
        # PPO-clip presets (ratio handles IS, so no separate IS weights needed):
        config = RolloutCorrectionConfig.bypass_ppo_clip()              # PPO-clip only
        config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()       # PPO-clip + Geo-RS
        config = RolloutCorrectionConfig.bypass_ppo_clip_k3_rs()        # PPO-clip + K3-RS
        # REINFORCE presets (explicit IS weights):
        config = RolloutCorrectionConfig.bypass_pg_is()                 # REINFORCE + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs()             # REINFORCE + Geo-RS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_seq_tis()     # REINFORCE + Geo-RS + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_token_tis()   # REINFORCE + Geo-RS + Token-TIS

        # Decoupled Geometric ratio presets (length-normalized IS ratio)
        config = RolloutCorrectionConfig.decoupled_geo_rs_seq_tis()           # Decoupled Geo-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_geo_rs_token_tis()         # Decoupled Geo-RS + Token-TIS

        # Decoupled K3 KL Estimator presets (more stable for small KL values)
        config = RolloutCorrectionConfig.decoupled_k3_rs()                    # Decoupled K3-RS
        config = RolloutCorrectionConfig.decoupled_k3_rs_seq_tis()            # Decoupled K3-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_k3_rs_token_tis()          # Decoupled K3-RS + Token-TIS

    Reference:
        Liu, Li, Fu, Wang, Liu, Shen (2025)
        "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
        https://richardli.xyz/rl-collapse
    """

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_is_batch_normalize: bool = False
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[str | float] = None
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Token-level Importance Sampling.

        IS weight correction at token level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with token-level IS
        """
        return cls(rollout_is="token", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level Importance Sampling.

        IS weight correction at sequence level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence-level IS
        """
        return cls(rollout_is="sequence", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is_rs(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.5_2.0",
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level IS + Rejection Sampling.

        Sequence-level IS with sequence-level rejection sampling in decoupled mode.
        Rejects entire sequences based on sequence-level IS weight.

        Args:
            is_threshold (float): Upper threshold for IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Upper threshold for rejection sampling. Default: 0.5_2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence IS + RS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_sum_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Geometric Mean Rejection Sampling (ratio-based).

        Uses geometric mean IS ratio E[log(r)] for rejection sampling at sequence level.
        This is a ratio-based mode (ideal = 0.0) with [lower, upper] threshold bounds.
        Length-normalized but still uses IS ratio semantics.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_ppo_clip(cls) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss.

        PPO clipped objective in bypass mode. The PPO ratio = π_θ/π_rollout
        already handles IS correction, so no explicit IS weights are applied.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip
        """
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and Geometric Mean RS (ratio-based).

        PPO clipped objective in bypass mode with geometric mean IS ratio RS.
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and K3 Rejection Sampling.

        PPO clipped objective in bypass mode with K3 KL estimator RS to mask outliers.
        K3 is more stable than K1 for small KL values.
        The PPO ratio = π_θ/π_rollout already handles IS correction.

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + K3-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and IS Correction.

        Uses REINFORCE loss with explicit IS correction in bypass mode.
        No PPO clipping.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + IS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and Geometric Mean RS (ratio-based).

        REINFORCE with geometric mean IS ratio rejection sampling in bypass mode.
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Sequence-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Clipped Sequence Weight (debiasing). Uses E[log(r)] (ideal = 0.0).

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Token-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Token-level IS weights. Uses E[log(r)] (ideal = 0.0).

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_pg_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Sequence-level IS.

        Combines geometric mean IS ratio rejection with sequence-level IS
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Token-level IS.

        Combines geometric mean IS ratio rejection with token-level IS weights
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses E[log(r)] (ideal = 0.0) with [lower, upper] threshold bounds.

        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (Optional[Union[str, float]]): Geometric RS threshold (upper). Default: 0.999_1.001 (±0.1%)

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 KL Estimator Rejection Sampling.

        Uses K3 KL estimator at sequence level for rejection sampling.
        K3 = E[r - log(r) - 1] where r = π_train/π_rollout.
        More stable than geometric mean for small KL values.

        K3 >= 0 always (equals 0 when policies match exactly).

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
                Typical range: 0.001-0.1

        Returns:
            RolloutCorrectionConfig configured for K3 RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Sequence-level Truncated IS.

        Combines K3 KL estimator rejection with sequence-level IS weights.
        K3 provides more stable outlier detection than geometric mean.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Token-level Truncated IS.

        Combines K3 KL estimator rejection with token-level IS weights.
        K3 provides more stable outlier detection than geometric mean.
        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        """Disabled - Metrics Only Mode.

        Computes and logs off-policy metrics without applying correction.

        Returns:
            RolloutCorrectionConfig with all correction disabled
        """
        return cls(rollout_is=None, rollout_rs=None)


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        length_adaptive_gae_alpha (float): Scale in
            ``lambda_i = 1 - 1 / max(alpha * response_length_i, 1)``.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        step_split (StepSplitConfig): Reusable reasoning-step boundary configuration.
        step_value (StepValueAdvantageConfig): Step-level advantage construction configuration.
        dapo_reference_kl (DapoReferenceKLConfig): Same-group correct-response KL auxiliary loss.
        ratio_value_critic (dict[str, Any]): Optimizer and initialization settings for the lightweight
            prefix-ratio value critic used when GAE is enabled without the standard critic worker.
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_correction (Optional[RolloutCorrectionConfig]): Rollout Correction configuration.
            Addresses off-policy issues from policy mismatch, model staleness, and general distribution shifts.

            Set to None to disable entirely. Use factory methods for common presets:
            - RolloutCorrectionConfig.decoupled_token_is() - Decoupled mode with token-level IS
            - RolloutCorrectionConfig.decoupled_seq_is() - Decoupled mode with sequence-level IS
            - RolloutCorrectionConfig.decoupled_seq_is_rs() - Decoupled mode with sequence IS + RS
            - RolloutCorrectionConfig.decoupled_k1_rs() - Decoupled mode with K1-RS (divergence)
            - RolloutCorrectionConfig.decoupled_geo_rs() - Decoupled mode with Geo-RS (ratio)
            - RolloutCorrectionConfig.bypass_ppo_clip() - Bypass mode with PPO-clip
            - RolloutCorrectionConfig.bypass_ppo_clip_k1_rs() - Bypass mode with PPO-clip + K1-RS
            - RolloutCorrectionConfig.bypass_pg_is() - Bypass mode with REINFORCE + IS
            - RolloutCorrectionConfig.bypass_pg_k1_rs() - Bypass mode with REINFORCE + K1-RS

            For backward compatibility, you can still pass a dict, which will be converted to
            RolloutCorrectionConfig automatically.
    """

    gamma: float = 1.0
    lam: float = 1.0
    length_adaptive_gae_alpha: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    step_split: StepSplitConfig = field(default_factory=StepSplitConfig)
    step_value: StepValueAdvantageConfig = field(default_factory=StepValueAdvantageConfig)
    dapo_reference_kl: DapoReferenceKLConfig = field(default_factory=DapoReferenceKLConfig)
    ratio_value_critic: dict[str, Any] = field(
        default_factory=lambda: {
            "a_init": 1.0,
            "b_init": 0.0,
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "update_steps": 1,
        }
    )
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Correction: corrects off-policy issues (policy mismatch, model staleness, distribution shifts)
    # Set to None to disable, use RolloutCorrectionConfig presets (e.g., .tis(), .mis()), or pass dict
    rollout_correction: Optional[RolloutCorrectionConfig] = None
    # GDPO (Group reward-Decoupled Normalization Policy Optimization) settings.
    # gdpo_reward_keys: keys in non_tensor_batch (from compute_score's return dict) that
    #   correspond to individual reward dimensions, e.g. ["format_reward", "accuracy_reward"].
    # gdpo_reward_weights: per-dimension weights for aggregation (default: equal weights).
    gdpo_reward_keys: Optional[list[str]] = None
    gdpo_reward_weights: Optional[list[float]] = None
