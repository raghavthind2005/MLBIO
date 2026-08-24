"""The `ca21` config block, as a typed extension of verl's PPOConfig.

WHY A DATACLASS AND NOT A LOOSE KEY. `verl/trainer/main.py:93` builds the config with
``OmegaConf.structured(PPOConfig())``, and a structured OmegaConf config is STRICT: an
unknown `ca21:` key in the YAML raises rather than being carried along. Attaching it by
``OmegaConf.set_struct(False)`` would work and would also disable the type checking that
catches `lam: "1.0"` (a string) or a misspelled `g_c`. Since the caption term IS the
method, its parameters get the same schema discipline as verl's own.

Every field here is a decision recorded in docs/DECISION_LOG.md. Defaults are deliberately
NOT the settled values: the config file states them explicitly so that a run's parameters
are visible in the artifact rather than inherited from code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CA21Config:
    """Parameters of the caption-distortion term."""

    #: O6 D3. Weight on J_cap relative to J_success. `caption_advantage` group-normalises
    #: D to z-scores, so this is scale-free -- lam=1.0 is genuine parity with the answer
    #: term regardless of D's raw range (0.007-0.786 across items in T0b).
    #: NOTE the effective value is 2-5% higher than nominal; see DECISION_LOG 4.15, where
    #: that bias is quantified and explicitly ACCEPTED rather than corrected.
    lam: float = 1.0

    #: O6 D4. Captions sampled per prompt. Must be a multiple of world_size or the composed
    #: training batch stops dividing evenly on some steps (protocol.py:555 asserts it, and
    #: the assert in ca21_trainer.fit explains why padding is not an acceptable fix).
    g_c: int = 8

    #: S13. Cap on shared trajectories per prompt used to score its captions. Trajectories
    #: are free -- J_success generated them anyway -- so this bounds only the scoring passes.
    m: int = 2

    #: S13/O4. Score captions only against trajectories that got the answer RIGHT. Under
    #: this gate the caption term does not reach prompts where J_success is dead: all-wrong
    #: groups have no correct trajectory and drop out here too (DECISION_LOG, _select_
    #: trajectories). It rescues the all-correct 4.7%, not the full 25%.
    correctness_gate: bool = True

    #: Rows per chunk in compute_caption_distortion. NOT a free knob: forward_packed_logits
    #: materialises [total_nnz, V] logits with V=152k, so the unchunked pass needs ~155 GB
    #: against 95 GiB on a GH200. T0e showed the packed forward is row-independent, so this
    #: cannot change the result beyond bf16 noise -- it is purely a memory/throughput dial.
    row_chunk: int = 16

    def post_init(self):
        if self.g_c < 2:
            raise ValueError(
                f"g_c={self.g_c}: a group of one has no within-group comparison, so S12's "
                f"advantage is identically zero and the caption term does nothing.")
        if self.m < 1:
            raise ValueError(f"m={self.m} must be >= 1")
        if self.row_chunk < 1:
            raise ValueError(f"row_chunk={self.row_chunk} must be >= 1")
        if self.lam < 0:
            raise ValueError(
                f"lam={self.lam} < 0 would REWARD captions that maximally distort the "
                f"policy, which is the objective backwards.")


def make_ca21_ppo_config(ppo_config_cls):
    """Subclass verl's PPOConfig with the `ca21` block.

    Factored like the other make_* helpers so this module imports without verl present,
    which is what lets the config be unit-tested off-cluster.
    """

    @dataclass
    class CA21PPOConfig(ppo_config_cls):
        ca21: CA21Config = field(default_factory=CA21Config)

    return CA21PPOConfig
