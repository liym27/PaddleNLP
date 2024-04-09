# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.utils import print_program_with_dist_attr
from tqdm.auto import tqdm

from paddlenlp.trainer import Trainer

from ..utils.import_utils import is_paddle_cuda_available
from ..utils.log import logger
from .argparser import strtobool
from .trainer import SCALER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from .trainer_callback import TrainerState
from .trainer_utils import (  # set_hyrbid_parallel_seed,
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    _exec_mode_guard,
    get_last_checkpoint,
    has_length,
    speed_metrics,
)
from .utils.helper import distributed_file, distributed_isfile  # nested_truncate,

try:
    from ..quantization.quantization_linear import QuantizationLinear
except:
    QuantizationLinear = None

MODEL_NAME = "model"
OPTIMIZER_NAME = "optimizer"
DIST_CKPT_PATH = "dist_ckpt"

# 2: check convergence
# 3: load and check accuracy
# 4: load but not check accuracy
RUN_MODE = 3

RESET_INPUTS = False

if RUN_MODE == 2:
    RESET_INPUTS = False
elif RUN_MODE == 3:
    RESET_INPUTS = True
else:
    RESET_INPUTS = False

# TAG = "single"
# TAG = "dp2"
# TAG = "mp2"
# TAG = "dp2mp2"
TAG = "dp2mp2pp2"
FILE_PREFIX = "auto_" + TAG


def log_loss(logs, file):
    loss = logs["loss"]
    global_step = logs["global_step"]
    learning_rate = logs["learning_rate"]
    # logwriter.add_scalar(tag=f"{TAG}/loss", value=loss, step=global_step)
    # logwriter.add_scalar(tag=f"{TAG}/learning_rate", value=learning_rate, step=global_step)
    # print(f"step {global_step} loss: {loss} md5: {loss._md5sum()} learning_rate: {learning_rate}")
    file.write(f"step {global_step} loss: {loss} learning_rate: {learning_rate}\n")
    file.flush()


def reset_inputs(inputs):
    inputs["input_ids"] = paddle.randint(0, 10000, (1, 1024))
    inputs["labels"] = paddle.randint(0, 10000, (1, 1024))
    return inputs


def print_inputs(inputs, step):
    input_ids, labels = inputs["input_ids"], inputs["labels"]
    # print(f"\n[print_inputs global] step {step} input_ids: {input_ids.shape} {input_ids._md5sum()}, labels: {labels._md5sum()}")
    print(
        f"[print_inputs] step {step} input_ids: {input_ids._local_value().shape} {input_ids._local_value()._md5sum()}, labels: {labels._local_value()._md5sum()}"
    )


def print_grad(model):
    print(f"-------------------print_grad------------------------------")
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    for p in model.parameters():
        assert p.name in name_mapping
        if p.grad is not None:
            print(f"[print_grad] {name_mapping[p.name]} {p.name}_grad {p.grad.shape} {p.grad._md5sum()}")
    print(f"----------------------print_grad auto end -----------------------\n")


def print_opt_param(state_dict):
    print(f"-------------------print_opt_param------------------------------")
    for k, v in state_dict.items():
        if isinstance(v, dict):
            print(f"[opt_param dict] {k} {len(v)}")
            for k1, k2 in v.items():
                print(f"  [opt_param dict] {k} {k1} {k2}")
        else:
            print(f"[opt_param local ] {k} {v._local_value().shape} {v._local_value()._md5sum()}")
            print(f"[opt_param global] {k} {v.shape} {v._md5sum()}")
    print(
        f"----------------------print_opt_param auto end ----count = {len(state_dict.items())}------------------------\n"
    )


def print_param(model):
    print(f"-------------------------------------------------")
    model_state_dict = model.state_dict()
    name_mapping = {v.name: k for (k, v) in model_state_dict.items()}
    count = 0
    for p in model.parameters():
        count += 1
        print(f"[print_param]{name_mapping[p.name]} {p.name} {p.shape} {p._md5sum()}")
    print(f"---------------print_param end-------auto----count = {count}------------------------\n")


class AutoTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        if kwargs.get("args", None) is not None and kwargs["args"].to_static:
            if kwargs.get("criterion", None) is None:

                def loss_func(loss, outputs):
                    return loss

                kwargs.update({"criterion": loss_func})

        super().__init__(*args, **kwargs)
        assert self.args.enable_auto_parallel

        self.global_mesh = fleet.auto.get_mesh()
        self.comm_group_in_pp = fleet.get_hybrid_communicate_group().get_pipe_parallel_group()

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        with _exec_mode_guard("dynamic"):
            if isinstance(tensors, paddle.Tensor):
                tr_loss = tensors._local_value() if tensors.is_dist() else tensors
            else:
                tr_loss = paddle.to_tensor([tensors])

        if self.args.pipeline_parallel_degree <= 1:
            return super()._nested_gather(tr_loss)

        paddle.distributed.broadcast(tr_loss, src=self.comm_group_in_pp.ranks[-1], group=self.comm_group_in_pp)

        return super()._nested_gather(tr_loss)

    def _wrap_model(self, model, training=True):
        return model

    def _get_meshes_for_loader(self):
        def _get_mesh(pp_idx=0):
            return self.global_mesh.get_mesh_with_dim("pp")[pp_idx]

        # Note(lizhiyu): If the values returned by `DataLoader` don't have the format `[images, labels]`,
        # error may occurs here.
        meshes = []
        meshes.append(_get_mesh(0))
        if self.args.pipeline_parallel_degree > 1:
            meshes.append(_get_mesh(self.args.pipeline_parallel_degree - 1))
        return meshes

    def _wrap_for_dist_loader(self, train_dataloader):
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=self._get_meshes_for_loader(),
            shard_dims="dp",
        )
        return dist_loader

    def _wrap_for_auto(self, model, train_dataloader):
        dist_loader = self._wrap_for_dist_loader(train_dataloader)

        if self.args.to_static:
            unified_strategy = dist.Strategy()
            unified_strategy._from_legacy_strategy(self.args.strategy)
            return (
                dist.to_static(model, dist_loader, self.criterion, self.optimizer, strategy=unified_strategy),
                dist_loader,
            )
        else:
            self.optimizer = dist.shard_optimizer(self.optimizer)
            return model, dist_loader

    def _wrap_amp_model(self, args, model):
        logger.info("Using half precision")
        if args.to_static:
            return
        self.enable_autocast_context_manager = True
        self.do_grad_scaling = True if self.args.fp16 else False
        self.amp_dtype = "float16" if self.args.fp16 else "bfloat16"
        self.scaler = dist.shard_scaler(paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss))
        if self.args.fp16_opt_level == "O2":
            paddle.amp.decorate(
                models=model,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                master_grad=self.args.amp_master_grad,
                excluded_layers=QuantizationLinear,
            )

    def _get_item_from_loss(self, loss):
        if isinstance(loss, paddle.Tensor):
            if loss.is_dist():
                return loss._local_value().item() if loss._is_initialized() else 0.0
            else:
                return loss.item() if loss._is_initialized() else 0.0
        else:
            return loss

    def _split_batches_for_accumulation(self, inputs):
        if self.args.gradient_accumulation_steps == 1:
            return [inputs]

        # if self.args.to_static:
        if self.args.to_static and self.args.pipeline_parallel_degree > 1:
            return [inputs]

        local_batches = [{} for i in range(self.args.gradient_accumulation_steps)]

        for key, value in inputs.items():
            ori_mesh, ori_placements = value.process_mesh, value.placements
            replicate_value = dist.reshard(value, ori_mesh, [dist.Replicate(), dist.Replicate()])
            local_datas = replicate_value.split(self.args.gradient_accumulation_steps, axis=0)

            for index, data in enumerate(local_datas):
                local_batches[index].update({key: dist.reshard(data, ori_mesh, ori_placements)})

        return local_batches

    def _inner_training_loop(
        self,
        args,
        model,
        train_dataloader,
        len_dataloader,
        max_steps,
        num_train_epochs,
        num_update_steps_per_epoch,
        num_train_samples,
        resume_from_checkpoint,
        ignore_keys_for_eval,
    ):
        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if (
            resume_from_checkpoint is not None
            and distributed_isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            and not self.args.ignore_load_lr_and_optim
        ):
            self.state = TrainerState.load_from_json(
                distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            )
            if self.args.world_size > 1:
                global_step_list = []
                paddle.distributed.all_gather(
                    global_step_list, paddle.to_tensor([self.state.global_step], dtype="int64")
                )
                assert (
                    paddle.sum(paddle.stack(global_step_list) - global_step_list[0]) == 0
                ), f"Error, get different globel step, please check! step list: {[x.item() for x in global_step_list]}"

            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        epoch_iterator = train_dataloader
        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        )
        if len_dataloader is not None:
            if self.args.gradient_accumulation_steps > len(epoch_iterator):
                logger.warning(
                    f"changing accumulation step from `{self.args.gradient_accumulation_steps}` to `{len(epoch_iterator)}` to avoid, cross epoch accumulate"
                )
                self.args.gradient_accumulation_steps = len(epoch_iterator)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        model, dist_loader = self._wrap_for_auto(model, train_dataloader)
        # if self.args.to_static:
        #     print_program_with_dist_attr(model.dist_main_program())
        train_dataloader = dist_loader()

        self.timers and self.timers("read-data").start()
        tt = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        hcg = fleet.get_hybrid_communicate_group()

        dp_degree = hcg.get_data_parallel_world_size()
        mp_degree = hcg.get_model_parallel_world_size()
        pp_degree = hcg.get_pipe_parallel_world_size()

        dp_rank = hcg.get_data_parallel_rank()
        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()

        # self.file = open(f"./records/{FILE_PREFIX}_{tt}.txt", "w")
        self.file = open(
            f"./records/auto_dp{dp_degree:02d}mp{mp_degree:02d}pp{pp_degree:02d}_tostatic_{self.args.to_static}_fp16_{self.args.fp16}_bf16_{self.args.bf16}_{tt}.txt",
            "w",
        )

        for epoch in range(epochs_trained, num_train_epochs):
            step_control = 0  # used in loop control, reset to 0 after every step
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            # read global-batch from dist_loader
            for step, inputs in enumerate(train_dataloader):
                # if step >= 1:
                #     exit(0)
                # print_inputs(inputs, step)
                # print_param(model)
                # print_grad(model)
                # if step >= 1:
                #     print_opt_param(self.optimizer.state_dict())

                self.timers and self.timers("read-data").stop()
                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

                # Skip past any already trained steps if resuming training
                # We use consumed_samples to reset the status
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                inputs_list = self._split_batches_for_accumulation(inputs)

                for inputs in inputs_list:
                    if step_control % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                        self.timers and self.timers("forward-backward").start()

                    tr_loss_step = self.training_step(model, inputs)

                    with _exec_mode_guard("dynamic"):
                        tr_loss += tr_loss_step
                        # logger.info(f"[YAMEI DEBUG] tr_loss : {tr_loss.numpy()}")
                        # logwriter.add_scalar("gpt_auto_hybrid_loss", value=tr_loss.numpy(), step=step)
                    disable_accumulation = self.args.pipeline_parallel_degree > 1 and self.args.to_static
                    # disable_accumulation = self.args.to_static

                    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                        or disable_accumulation
                    ):

                        self.timers and self.timers("forward-backward").stop()

                        self.timers and self.timers("optimizer-step").start()

                        # Optimizer step
                        self.callback_handler.on_optimizer_begin(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        self.optimizer_step()

                        self.timers and self.timers("optimizer-step").stop()

                        self.callback_handler.on_optimizer_end(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                        self._print_timer()
                        step_control = 0
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                        step_control += 1

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                self.timers and self.timers("read-data").start()

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break
        self.file.close()
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")

        self._total_loss_scalar += self._get_item_from_loss(tr_loss)
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        total_batch_size_per_acc_step = self.args.per_device_train_batch_size * self.args.dataset_world_size
        total_batch_size = total_batch_size_per_acc_step * self.args.gradient_accumulation_steps

        return paddle.io.BatchSampler(
            dataset=self.train_dataset,
            shuffle=False,
            batch_size=total_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        outputs = model(**inputs)
        # print(f"outputs: {outputs}")
        # print(
        #     f"[yamei] logits: local {outputs._local_value().shape}  {outputs._local_value()._md5sum()} global {outputs.shape} {outputs._md5sum()}"
        # )

        if self.criterion is not None:

            def to_list(value):
                if value is None:
                    return value
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [value]

            criterion_inputs = to_list(outputs)
            criterion_labels = to_list(labels)
            loss = self.criterion(*(criterion_inputs + criterion_labels))
            outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs

        return (loss, outputs) if return_outputs else loss

    def dynamic_traning(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if loss is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def static_traning(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        input_ids, labels = tuple(inputs.values())
        loss = model(input_ids, labels)

        if loss is not None and self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)

        if not self.args.to_static:
            loss = self.dynamic_traning(model, inputs)
        else:
            loss = self.static_traning(model, inputs)

        if isinstance(loss, paddle.Tensor):
            return loss.detach() if loss._is_initialized() else float(0.0)
        elif isinstance(loss, np.ndarray):
            return np.sum(loss)
        elif loss is None:
            return float(0.0)
        else:
            return float(loss)

    def optimizer_step(self):
        if not self.args.to_static:
            optimizer_was_run = True
            if self.do_grad_scaling:
                scale_before = paddle.assign(self.scaler._scale)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler._scale
                # Compatible with paddlepaddle 2.6.0 using typo word.
                if hasattr(self.scaler, "_cache_founf_inf"):
                    optimizer_was_run = not self.scaler._cache_founf_inf
                else:
                    optimizer_was_run = not self.scaler._cache_found_inf
                if not optimizer_was_run:
                    scale_before_value = scale_before.cpu().numpy()
                    scale_after_value = scale_after.cpu().numpy()
                    logger.warning(
                        f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                    )
            else:
                self.optimizer.step()

            if optimizer_was_run:
                self.lr_scheduler.step()

            self.optimizer.clear_grad()
        else:
            # TODO: support optimizer_was_run in static mode
            self.lr_scheduler.step()

    def _maybe_log_save_evaluate_back(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._get_item_from_loss(self._nested_gather(tr_loss).mean())

            # reset tr_loss to zero
            tr_loss.subtract_(tr_loss)

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
            logs["learning_rate"] = float("{0:.3e}".format(self._get_learning_rate()))
            logs["global_step"] = int(self.state.global_step)

            log_loss(logs, self.file)

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.dataset_world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            # Add additional memory in log.
            if not self.args.skip_memory_metrics:
                logs.update(
                    {
                        "cpu_mem_used": self._memory_tracker.cpu_mem_used() >> 20,
                        "cpu_mem_used_peak": self._memory_tracker.cpu_mem_used_peak >> 20,
                    }
                )
                if is_paddle_cuda_available():
                    logs.update(
                        {
                            "gpu_max_memory_allocated": paddle.device.cuda.max_memory_allocated() >> 20,
                            "gpu_max_memory_reserved": paddle.device.cuda.max_memory_reserved() >> 20,
                        }
                    )

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.optimizer, GroupShardedOptimizerStage2) and self.optimizer._broadcast_overlap:
                paddle.device.cuda.synchronize()

            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            if isinstance(self.optimizer, GroupShardedOptimizerStage2) and self.optimizer._broadcast_overlap:
                paddle.device.cuda.synchronize()

            self._save_checkpoint(model, metrics=metrics)
            logger.info(f"{self.runtime_timer.log()}")
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        with _exec_mode_guard("dynamic"):
            self._maybe_log_save_evaluate_back(tr_loss, model, epoch, ignore_keys_for_eval, **kwargs)

    def _save_checkpoint(self, model, metrics=None):

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = f"{run_dir}/{checkpoint_folder}"

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            logger.info(f"Saving checkpoinit files into {output_dir}")

            if self.args.should_save_model_state:

                optim_state_dict = self.optimizer.state_dict()
                optim_state_dict.pop("LR_Scheduler", None)

                state_dict = {
                    MODEL_NAME: self.model.state_dict(),
                    OPTIMIZER_NAME: optim_state_dict,
                }

                self._save_ckpt_func(state_dict, os.path.join(output_dir, DIST_CKPT_PATH))
                logger.info(f"Model weights and optimizer states saved in {output_dir}/{DIST_CKPT_PATH}")

                # FIXME: maybe only save one copy
                paddle.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

                if self.do_grad_scaling:
                    paddle.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": [k.current_seed() for k in paddle.get_rng_state()],
            "cpu": paddle.framework.core.default_cpu_generator().get_state().current_seed(),
        }
        # if self.args.use_hybrid_parallel:
        #     rng_states[
        #         "hybrid_parallel_rng_state_tracker"
        #     ] = fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()

        if self.args.world_size > 1:
            rng_states_list = []
            paddle.distributed.all_gather_object(rng_states_list, rng_states)
            if self.args.should_save:
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(rng_states_list, os.path.join(output_dir, f"rng_state_{self.args.world_size}.pth"))
        else:
            os.makedirs(output_dir, exist_ok=True)
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
            # save checkpoint_done file to ensure checkpoint is complete
            if self.args.should_save_model_state and self.args.should_save:
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".checkpoint_done"))

    def _save(self, output_dir: Optional[str] = None, state_dict=None, merge_tensor_parallel=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.should_save_model_state:
            self._save_ckpt_func(self.model.state_dict(), os.path.join(output_dir, MODEL_NAME))
            logger.info(f"Model weights saved in {output_dir}/{MODEL_NAME}")

    def _load_from_checkpoint(self, resume_from_checkpoint=None):

        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None:

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            if not self.args.ignore_load_lr_and_optim:
                with _exec_mode_guard("dynamic"):
                    if distributed_isfile(os.path.join(resume_from_checkpoint, SCHEDULER_NAME)):
                        self.lr_scheduler.set_state_dict(
                            paddle.load(distributed_file(os.path.join(resume_from_checkpoint, SCHEDULER_NAME)))
                        )
                    else:
                        raise ValueError(
                            f"scheduler-file not found, scheduler:{os.path.join(resume_from_checkpoint, SCHEDULER_NAME)}"
                        )

                    if self.do_grad_scaling and distributed_isfile(os.path.join(resume_from_checkpoint, SCALER_NAME)):
                        self.scaler.load_state_dict(
                            paddle.load(
                                distributed_file(os.path.join(resume_from_checkpoint, SCALER_NAME)), return_numpy=True
                            )
                        )

            ckpt_path = os.path.join(resume_from_checkpoint, DIST_CKPT_PATH)

            if not os.path.isdir(ckpt_path):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            optim_state_dict = self.optimizer.state_dict()
            optim_state_dict.pop("LR_Scheduler", None)

            state_dict = {
                MODEL_NAME: self.model.state_dict(),
                OPTIMIZER_NAME: optim_state_dict,
            }

            print("state_dict :", state_dict)

            self._load_ckpt_func(state_dict, ckpt_path)

            # release memory
            del state_dict
