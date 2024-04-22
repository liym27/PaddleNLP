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
# export PYTHONPATH="/paddle/Paddle/build_gpu/python":$PYTHONPATH
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1 
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_call_stack_level=3

to_static=1
export TRANSLATOR_DISABLE_NEW_ERROR=0
export TRANSLATOR_CODE_LEVEL=100

task_name="gpt3_auto_dp2mp2pp2_static_${to_static}"
log_dir="log/$task_name"
output_dir="output/$task_name"
rm -rf $log_dir
# rm -rf $output_dir

input_dir="../../data"

python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    ../run_pretrain_auto.py \
    --model_name_or_path gpt3-1.3B-en \
    --tokenizer_name_or_path gpt3-1.3B-en \
    --input_dir  ${input_dir}  \
    --output_dir ${output_dir}  \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --sequence_parallel 0 \
    --fp16 0 \
    --fp16_opt_level "O2"  \
    --recompute 1 \
    --recompute_granularity "core_attn" \
    --use_flash_attention 0 \
    --fuse_attention_qkv 0 \
    --sharding "" \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 30 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 0 \
    --logging_steps 10 \
    --continue_training 0\
    --dataloader_num_workers 1 \
    --eval_steps 100000 \
    --report_to "none" \
    --disable_tqdm true \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --model_type "gpt" \
    --enable_auto_parallel 1 \
    --to_static ${to_static} \
    --skip_memory_metrics 0 \