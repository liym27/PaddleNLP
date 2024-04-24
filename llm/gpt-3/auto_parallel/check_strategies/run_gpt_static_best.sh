# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

export PYTHONPATH="../../../../":$PYTHONPATH
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1 
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_call_stack_level=3
to_static=1
export TRANSLATOR_DISABLE_NEW_ERROR=0
export TRANSLATOR_CODE_LEVEL=100

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export PADDLE_NNODES=1

# ----------- 1. 模型配置 -----------
MODEL_TYPE="gpt3-13B-en"

# ----------- 2. 分布式配置 -----------
WORLD_SIZE=8
GBS=32
MBS=1
MP=2
PP=4
VPP=1
sharding="stage1"
DP=$(($WORLD_SIZE / ($MP * $PP)))
ACC_STEPS=$(($GBS / ($DP * $MBS)))

# ---------- 3. 优化策略配置 old ----------
sequence_parallel=1     # default: 1
use_flash_attention=1   # default: 1
fuse_attention_qkv=1    # default: 1
use_fused_dropout_add=1

recompute_args="--recompute 1 \
                --recompute_use_reentrant true \
                --recompute_granularity full \
                --pp_recompute_interval 1"
amp_args="--bf16 1 \
          --fp16_opt_level 'O2'"

# ---------- 4. 优化策略配置 new ----------
use_fast_layer_norm=1
use_fused_linear=1
use_fused_dropout_add=1
enable_linear_fused_grad_add=1
tensor_parallel_config="enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add"
sharding_parallel_config="enable_stage1_tensor_fusion enable_stage1_overlap"
if [ "$autoconfig_args" = "" ]; then
  #if [ "$MP" != "1" ]; then
  #  export CUDA_DEVICE_MAX_CONNECTIONS=1
  #fi
  if [ "$sequence_parallel" = "1" ]; then
    extra_pp_config="disable_partial_send_recv"
  fi
fi

# ----------- 5. 训练配置 -----------
max_steps=30        # default: 30
logging_steps=10    # default: 10

# -------------------------------

task_name="gpt3_perf_hand_best"
log_dir="log/$task_name"
output_dir="output/$task_name"
input_dir="../../data"

python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    ../run_pretrain_auto.py \
    --model_name_or_path "${MODEL_TYPE}.json" \
    --tokenizer_name_or_path ${MODEL_TYPE} \
    --input_dir  ${input_dir}  \
    --output_dir ${output_dir}  \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree ${MP} \
    --pipeline_parallel_degree ${PP} \
    --virtual_pp_degree ${VPP} \
    --sequence_parallel ${sequence_parallel} \
    ${recompute_args} \
    ${amp_args} \
    --use_flash_attention ${use_flash_attention} \
    --fuse_attention_qkv ${fuse_attention_qkv} \
    --use_fused_dropout_add ${use_fused_dropout_add} \
    --use_fast_layer_norm ${use_fast_layer_norm} \
    --use_fused_linear ${use_fused_linear} \
    --enable_linear_fused_grad_add ${enable_linear_fused_grad_add} \
    --sharding ${sharding} \
    --sharding_parallel_config ${sharding_parallel_config} \
    --tensor_parallel_config ${tensor_parallel_config} \
    --pipeline_parallel_config "enable_sharding_comm_overlap ${extra_pp_config}" \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps ${max_steps} \
    --logging_steps ${logging_steps} \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 0 \
    --continue_training 0\
    --dataloader_num_workers 1 \
    --eval_steps 100000 \
    --report_to "none" \
    --disable_tqdm true \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --do_train \
    --do_eval \
    --device "gpu" \
    --model_type "gpt" \
    --enable_auto_parallel 1 \
    --to_static ${to_static} \
    --skip_memory_metrics 0 \
