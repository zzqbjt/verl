# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
set -x
WORKING_DIR="/data2/zzn/verl"
RUNTIME_ENV="/data2/zzn/verl/examples/grpo_trainer/runtime_env.yaml"
max_prompt_length=512
max_response_length=4096
num_gpus=2
train_batch_size=128
mini_batch_size=16
group_size=8
bs_per_gpu=$((mini_batch_size/num_gpus))

loss_mode=my
tau=${TAU:-0.5}
decay_rate=32.0
chunk_size=128
future_kl_start=include_current
future_kl_window=-1
future_kl_average=False
future_kl_clip_ratio=0.2
future_kl_clip_high_only=True
safety_thresh=10.0


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=/data2/zzn/dataset/DAPO/dapo-math-17k.parquet \
        data.val_files=/data2/zzn/dataset/DAPO/aime-2024.parquet \
        data.train_batch_size=${train_batch_size} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/data2/zzn/models/Qwen/Qwen3-1.7B-Base \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${bs_per_gpu} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.calculate_entropy=True \
        actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
        actor_rollout_ref.actor.policy_loss.tau=${tau} \
        +actor_rollout_ref.actor.policy_loss.decay_rate=${decay_rate} \
        +actor_rollout_ref.actor.policy_loss.chunk_size=${chunk_size} \
        +actor_rollout_ref.actor.policy_loss.future_kl_start=${future_kl_start} \
        +actor_rollout_ref.actor.policy_loss.future_kl_window=${future_kl_window} \
        +actor_rollout_ref.actor.policy_loss.future_kl_average=${future_kl_average} \
        +actor_rollout_ref.actor.policy_loss.future_kl_clip_ratio=${future_kl_clip_ratio} \
        +actor_rollout_ref.actor.policy_loss.future_kl_clip_high_only=${future_kl_clip_high_only} \
        +actor_rollout_ref.actor.policy_loss.safety_thresh=${safety_thresh} \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${bs_per_gpu} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.n=${group_size} \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${bs_per_gpu} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.log_val_generations=1 \
        trainer.val_before_train=True \
        trainer.critic_warmup=0 \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name='RLVR' \
        trainer.experiment_name='Qwen3-1.7B-GRPO-my3' \
        trainer.n_gpus_per_node=${num_gpus} \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.default_local_dir=/data2/zzn/models/verl_ckpts/GRPO-my3/Qwen3-1.7B-Base \
        trainer.total_epochs=1 $@
        
