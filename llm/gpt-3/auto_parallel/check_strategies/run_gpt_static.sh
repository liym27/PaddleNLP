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

task_name="gpt3_static_mp2pp4_perf"
log_dir="log/$task_name"
output_dir="output/$task_name"

input_dir="../../data"
WORLD_SIZE=8
GBS=32
MBS=1
MP=2
SP=0  # 0 or 1
PP=4
VPP=1
SD=$(($WORLD_SIZE / ($MP * $PP)))
ACC_STEPS=$(($GBS / ($SD * $MBS)))
SEQLEN=4096
MODEL_TYPE="gpt3-1.3B-en"
recompute_args="--recompute 1 \
                --recompute_use_reentrant true \
                --recompute_granularity full \
                --pp_recompute_interval 1"

if [ "$autoconfig_args" = "" ]; then
  #if [ "$MP" != "1" ]; then
  #  export CUDA_DEVICE_MAX_CONNECTIONS=1
  #fi
  if [ "$SP" = "1" ]; then
    extra_pp_config="disable_partial_send_recv"
  fi
fi

python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    ../run_pretrain_auto.py \
    --model_name_or_path "${MODEL_TYPE}" \
    --tokenizer_name_or_path "${MODEL_TYPE}" \
    --input_dir  ${input_dir}  \
    --output_dir ${output_dir}  \
    --split 949,50,1 \
    --max_seq_length ${SEQLEN} \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --per_device_eval_batch_size 4 \
    --bf16 1 \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --tensor_parallel_degree ${MP} \
    --pipeline_parallel_degree ${PP} \
    --virtual_pp_degree ${VPP} \
    --sequence_parallel ${SP} \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --do_train \
    --max_steps 30 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --logging_steps 1 \
    ${recompute_args} \
    --dataloader_num_workers 1 \
    --use_flash_attention true \
    --use_fused_rms_norm true \
    --fuse_attention_qkv true \
    --use_fast_layer_norm true \
    --use_fused_linear false \
    --use_fused_dropout_add false \
    --use_fused_rope true \
    --enable_linear_fused_grad_add false \
    --sharding "stage1" \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --disable_tqdm true \
    --continue_training 0 \
    --skip_memory_metrics 0 \
    --report_to "none" \
    --model_type "gpt" \
    --enable_auto_parallel 1 \
    --to_static ${to_static} \
    --scale_loss 1024 \
    --device "gpu" 2>&1 | tee log_${OUTPUT_FILENAME}.txt
