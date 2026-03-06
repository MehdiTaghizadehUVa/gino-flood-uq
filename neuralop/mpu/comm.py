# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import torch
import torch.distributed as dist
import datetime as dt

class disable_logging(object):
    def __init__(self, level=logging.ERROR):
        logging.disable(level=level)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        logging.disable(level=logging.NOTSET)


# dummy placeholders
_DATA_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None
_LOCAL_RANK = 0
_GLOBAL_RANK = 0
_WORLD_SIZE = 1

# world comm
def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return _WORLD_SIZE


def get_local_rank():
    if dist.is_initialized():
        return _LOCAL_RANK
    return int(os.getenv("LOCAL_RANK", 0))


def get_global_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.getenv("RANK", 0))

# data parallel
def get_data_parallel_size():
    if not dist.is_initialized():
        return 1
    if _DATA_PARALLEL_GROUP is None:
        return dist.get_world_size()
    return dist.get_world_size(group=_DATA_PARALLEL_GROUP)


def get_data_parallel_rank():
    if not dist.is_initialized():
        return 0
    if _DATA_PARALLEL_GROUP is None:
        return dist.get_rank()
    return dist.get_rank(group=_DATA_PARALLEL_GROUP)

def get_data_parallel_group():
    assert dist.is_initialized(), "Error, initialize torch.distributed first"
    return _DATA_PARALLEL_GROUP 


# model parallel
def get_model_parallel_size():
    if not dist.is_initialized() or (_MODEL_PARALLEL_GROUP is None):
        return 1
    return dist.get_world_size(group=_MODEL_PARALLEL_GROUP)


def get_model_parallel_rank():
    if not dist.is_initialized() or (_MODEL_PARALLEL_GROUP is None):
        return 0
    return dist.get_rank(group=_MODEL_PARALLEL_GROUP)


def get_model_parallel_group():
    assert dist.is_initialized(), "Error, initialize torch.distributed first"
    return _MODEL_PARALLEL_GROUP  


def init(model_parallel_size: int=1, verbose: bool=False):
    """
    Set up global and local communicator.
    `torchrun` initializes rank env vars by default and uses its own wireup logic
    """

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    global_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    timeout_min = int(os.getenv("TORCH_DISTRIBUTED_TIMEOUT_MIN", "30"))

    if world_size > 1:
        with disable_logging():
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    rank=global_rank,
                    world_size=world_size,
                    timeout=dt.timedelta(minutes=timeout_min),
                )
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    global _LOCAL_RANK
    global _GLOBAL_RANK
    global _WORLD_SIZE
    _LOCAL_RANK = local_rank
    _GLOBAL_RANK = global_rank
    _WORLD_SIZE = world_size

    # process 0 is logger
    is_logger = (global_rank == 0)

    # get model groups
    model_group_size = model_parallel_size
    
    # compute data parallel size 
    data_group_size = world_size // model_group_size

    if is_logger:
        print(f"Using {world_size} in {model_group_size} x {data_group_size} decomposition (#model-ranks x #data-ranks)")

    assert ( (model_group_size <= world_size) and (world_size % model_group_size == 0) ), \
        "Error, please make sure matmul_parallel_size * spatial_parallel_size <= world size and that world size is evenly divisible by matmul_parallel_size * spatial_parallel_size"
    
    # number of model groups
    num_model_groups = world_size // model_group_size

    global _DATA_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP

    if is_logger:
        print("Starting Wireup")

    if world_size > 1:
        if model_group_size > 1:
            model_groups = []
            for i in range(num_model_groups):
                start = i*model_group_size
                end = start + model_group_size
                model_groups.append(list(range(start, end)))
                    
            data_groups = [sorted(list(i)) for i in zip(*model_groups)]                     

            if verbose and is_logger:
                print("Model Parallel Groups w/ respect to world rank:")
                for grp in model_groups:
                    print(grp)
            
            if verbose and is_logger:
                print("Data Parallel Groups w/ respect to world rank:")
                for grp in data_groups:
                    print(grp)

            # initialize groups
            with disable_logging():
                # data groups
                for grp in data_groups:
                    tmp_group = dist.new_group(ranks = grp)
                    if global_rank in grp:
                        _DATA_PARALLEL_GROUP = tmp_group
                # model groups
                for grp in model_groups:
                    tmp_group = dist.new_group(ranks = grp)
                    if global_rank in grp:
                        _MODEL_PARALLEL_GROUP = tmp_group
                                
        else:
            # Single model-parallel shard: one data-parallel group spanning all ranks.
            with disable_logging():
                _DATA_PARALLEL_GROUP = dist.new_group(ranks = list(range(world_size)))
                _MODEL_PARALLEL_GROUP = None

    # barrier
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    if is_logger:
        print("Finished Wireup")
    
    return
