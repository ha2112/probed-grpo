#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/env.sh"
dry_run=false
config_only=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
if [[ "${1:-}" == --config-only ]]; then config_only=true; shift; fi
route="${1:-}"
if [[ "${route}" != random && "${route}" != official && "${route}" != probed ]]; then
    echo "Usage: $0 [--dry-run|--config-only] {random|official|probed} [additional verl overrides...]" >&2
    exit 2
fi
shift

DATA_DIR="${CODECONTESTS_GRPO_DATA_DIR:-${SCRIPT_DIR}/data}"
MODEL_PATH="${AFTERBURNER_MODEL_PATH:-${ROOT_DIR}/model-cache/afterburner/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start}"
CHECKPOINT_DIR="${AFTERBURNER_CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
train_file="${DATA_DIR}/${route}/train.parquet"
validation_file="${DATA_DIR}/shared/validation.parquet"

if [[ "${dry_run}" == false && "${config_only}" == false ]] && [[ ! -f "${train_file}" || ! -f "${validation_file}" ]]; then
    echo "Missing prepared corpus under ${DATA_DIR}; run codeforces_corpus.py first." >&2
    exit 1
fi

# Keep Hub downloads in this repository instead of the user's global cache.
if [[ "${dry_run}" == false && "${config_only}" == false ]]; then
    MODEL_PATH="$("${PYTHON_BIN}" "${ROOT_DIR}/artifact_cache.py" "${MODEL_PATH}")"
fi

overrides=( \
    algorithm.adv_estimator=grpo \
    data.train_files="['${train_file}']" \
    data.val_files="['${validation_file}']" \
    data.shuffle=False \
    data.validation_shuffle=False \
    +data.seed="${EXPERIMENT_SEED}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    "custom_reward_function.path='${SCRIPT_DIR}/codeforces_reward.py'" \
    custom_reward_function.name=codeforces_reward_fn_batch \
    reward_model.reward_manager=batch \
    "actor_rollout_ref.model.path='${MODEL_PATH}'" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.seed="${EXPERIMENT_SEED}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.balance_batch=False \
    trainer.project_name=verl_grpo_afterburner_codecontests \
    trainer.experiment_name="codecontests-${route}-seed-${EXPERIMENT_SEED}" \
    "trainer.default_local_dir='${CHECKPOINT_DIR}/${route}-seed-${EXPERIMENT_SEED}'" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs="${TOTAL_EPOCHS}" "$@" )

if [[ "${dry_run}" == true ]]; then
    printf '%q ' "${PYTHON_BIN}" -m verl.trainer.main_ppo "${overrides[@]}"
    printf '\n'
    exit 0
fi
if [[ "${config_only}" == true ]]; then
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" --stage config -- "${overrides[@]}"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" --stage training \
    --record "${ROOT_DIR}/results/launch-${route}.yaml" -- "${overrides[@]}"
exec "${PYTHON_BIN}" -m verl.trainer.main_ppo "${overrides[@]}"
