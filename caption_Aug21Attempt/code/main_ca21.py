"""Training entry point. verl/trainer/main.py with three substitutions and nothing else.

TRANSCRIBED, like ca21_trainer.fit, and for the same reason: there is no extension point.
The substitutions are:

    RLHFDataset        -> make_ca21_dataset(RLHFDataset)      carries the question text
    FSDPWorker         -> make_ca21_worker(FSDPWorker, ...)   adds compute_caption_distortion
    RayPPOTrainer      -> make_ca21_trainer(RayPPOTrainer)    adds the caption term
    PPOConfig          -> make_ca21_ppo_config(PPOConfig)     adds the `ca21` block

Everything else -- Ray init, runtime env, resource pools, reward managers, dataloaders --
is upstream verbatim so that an upstream change is a diff rather than a divergence.

WHY THE DATASET IS SWAPPED BY REBINDING rather than by editing create_dataloader:
`data_loader.py:27,68` constructs RLHFDataset directly, and create_dataloader also handles
filtering, samplers and the val split. Reimplementing it to change one class would be
exactly the kind of faithful-looking port that drifts from upstream. Rebinding the name in
that module's namespace changes the one thing we need and leaves the rest untouched.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ray                                                              # noqa: E402
from omegaconf import OmegaConf                                         # noqa: E402

#: Pinned by T0e inside the container (DECISION_LOG 4.18). ca21_trainer.fit hashes verl's
#: own fit() at startup and refuses to run if upstream has moved, because this fork was
#: derived line-by-line from that version.
EXPECTED_FIT_SHA256 = "3c884df495e3f95f6711f83311f99233389b79fbdaf1f5d379874f506102e49d"


def _assert_format_prompt_parity(format_prompt_path):
    """The dataset's rendered prompt must equal ``shared_tail`` byte for byte.

    Rendered with the SAME machinery verl uses (jinja2 Template on the stripped file,
    ``content=`` the problem) so this compares what will actually run, not a re-derivation
    of it -- the mistake that produced four false alarms in the T0d/T0e stretch.
    """
    from jinja2 import Template

    from ca21_prompts import shared_tail

    if not format_prompt_path:
        raise ValueError(
            "data.format_prompt is unset, so the rollout prompt would be the bare question "
            "while the blind pass appends the shared instruction. The two scored contexts "
            "would differ by more than the evidence.")
    text = Path(format_prompt_path).read_text(encoding="utf-8")
    probes = ["What is 2+2?", "  How many apples are on the table?  ",
              "Solve for x:\n2x = 4"]
    for probe in probes:
        rendered = Template(text.strip()).render(content=probe)
        want = shared_tail(probe)
        if rendered != want:
            raise AssertionError(
                f"data.format_prompt={format_prompt_path} does not reproduce "
                f"ca21_prompts.shared_tail.\n  rendered: {rendered!r}\n  shared_tail: "
                f"{want!r}\nG-PARITY requires them identical: the sighted context is the "
                f"rendered rollout prompt and the blind context is shared_tail, so any "
                f"difference is scored as if it were the caption's doing.")
    print(f"[ca21] G-PARITY ok: {format_prompt_path} == shared_tail", flush=True)


@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config):
        from verl.single_controller.base.decorator import Dispatch, register
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer import data_loader as _data_loader
        from verl.trainer.data_loader import create_dataloader
        from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils.dataset import RLHFDataset
        from verl.utils.tokenizer import get_processor, get_tokenizer
        from verl.workers.fsdp_workers import FSDPWorker
        from verl.workers.reward import AutoRewardManager

        from ca21_dataset import make_ca21_dataset
        from ca21_trainer import make_ca21_trainer
        from ca21_worker import make_ca21_worker

        print(json.dumps(config.to_dict(), indent=2))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # --- SUBSTITUTION 1: the dataset carries the question text -------------------
        # Rebound in data_loader's namespace, which is where create_dataloader looks it up.
        _data_loader.RLHFDataset = make_ca21_dataset(RLHFDataset)
        print(f"[ca21] dataset -> {_data_loader.RLHFDataset.__name__}", flush=True)

        # --- SUBSTITUTION 2: the worker can score captions ---------------------------
        CA21Worker = make_ca21_worker(FSDPWorker, register, Dispatch.DP_COMPUTE_PROTO)
        print(f"[ca21] worker  -> {CA21Worker.__name__}", flush=True)

        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(CA21Worker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(
            num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(
            config.data, tokenizer, processor)

        # --- SUBSTITUTION 3: the trainer runs the caption term ------------------------
        CA21Trainer = make_ca21_trainer(
            RayPPOTrainer, expected_fit_sha256=EXPECTED_FIT_SHA256)
        print(f"[ca21] trainer -> {CA21Trainer.__name__} "
              f"(fit pinned {EXPECTED_FIT_SHA256[:12]}...)", flush=True)

        trainer = CA21Trainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    from verl.trainer.config import PPOConfig

    from ca21_config import make_ca21_ppo_config

    cli_args = OmegaConf.from_cli()
    CA21PPOConfig = make_ca21_ppo_config(PPOConfig)
    default_config = OmegaConf.structured(CA21PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()
    # recursive_post_init walks verl's own tree; ours is a sibling field, so call it here.
    ppo_config.ca21.post_init()

    # The caption term is the whole method. A config that silently lost its `ca21` block
    # would train Arm A under Arm B's name and log it under Arm B's run id.
    if ppo_config.ca21.g_c % (ppo_config.trainer.n_gpus_per_node * ppo_config.trainer.nnodes):
        raise ValueError(
            f"g_c={ppo_config.ca21.g_c} is not a multiple of world_size "
            f"{ppo_config.trainer.n_gpus_per_node * ppo_config.trainer.nnodes}. The composed "
            f"answer+caption batch would stop dividing evenly on whichever step first "
            f"produced an awkward surviving-prompt count -- hours in. Checked here, before "
            f"a GPU is allocated.")

    # G-PARITY, checked before a GPU is allocated.
    #
    # THE FAILURE THIS PREVENTS. The sighted context scored by S11 is the ACTUAL rollout
    # prompt, which verl builds by rendering data.format_prompt (dataset.py:157-161). The
    # blind context is built here by ca21_prompts.shared_tail. If those two disagree, the
    # two scored contexts differ by MORE than the evidence, and `D` measures the prompt
    # mismatch instead of what the caption preserved -- finite, plausible, and wrong.
    # T0a/T0b never saw this because they called build_sighted_messages directly; the
    # divergence only appears in production, where the prompt comes from the dataset.
    _assert_format_prompt_parity(ppo_config.data.format_prompt)

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                # NEVER expandable_segments:True -- it crashes the vLLM CuMemAllocator on
                # this stack (learned in the PAPO line; see runs/_env.sh).
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                # code/ is a plain cp target, not on the image's path -- but this must
                # EXTEND the inherited PYTHONPATH, not replace it. _env.sh puts
                # EasyR1_ca21 there, and a Ray actor started with only code/ on its path
                # dies with "No module named 'verl'" (job 3175577) -- inside the actor, so
                # the traceback arrives wrapped in RayTaskError rather than at the point
                # of the mistake.
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(Path(__file__).resolve().parent),
                                os.environ.get("PYTHONPATH", "")) if p),
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
