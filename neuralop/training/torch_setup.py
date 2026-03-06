import torch
import torch.distributed as dist
import neuralop.mpu.comm as comm

def _safe_get(obj, key, default):
    try:
        return getattr(obj, key)
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        return default


def setup(config):
    """A convenience function to intialize the device, setup torch settings and
    check multi-grid and other values. It sets up distributed communitation, if used.
    
    Parameters
    ----------
    config : dict 
        this function checks:
        * config.distributed (use_distributed, seed)
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)
    
    Returns
    -------
    device, is_logger
        device : torch.device
        is_logger : bool
    """
    use_distributed = bool(_safe_get(config.distributed, "use_distributed", False))
    if use_distributed:
        verbose = bool(_safe_get(config, "verbose", False))
        comm.init(model_parallel_size=config.distributed.get('model_parallel_size', 1),
                  verbose=verbose)

        # Set global rank 0 to log screen and wandb.
        is_logger = (comm.get_global_rank() == 0)

        # Set device and random seed.
        device = torch.device(f"cuda:{comm.get_local_rank()}")
        seed = config.distributed.seed + comm.get_data_parallel_rank()

        #Ensure batch can be evenly split among the model-parallel group
        patching = _safe_get(config, "patching", None)
        patching_levels = int(_safe_get(patching, "levels", 0)) if patching is not None else 0
        if patching_levels > 0:
            assert(config.data.batch_size*(2**(2*config.patching.levels)) % comm.get_model_parallel_size() == 0), (
                f'With MG patching, total batch-size of {config.data.batch_size*(2**(2*config.patching.levels))}'
                f' ({config.data.batch_size} times {(2**(2*config.patching.levels))}).'
                f' However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups.'
            )
            for j, b_size in enumerate(config.data.test_batch_sizes):
                assert (b_size*(2**(2*config.patching.levels)) % comm.get_model_parallel_size() == 0), (
                f'With MG patching, for test resolution of {config.data.test_resolutions[j]}'
                f' the total batch-size is {config.data.batch_size*(2**(2*config.patching.levels))}'
                f' ({config.data.batch_size} times {(2**(2*config.patching.levels))}).'
                f' However, this total batch-size cannot be evenly split among the {comm.get_model_parallel_size()} model-parallel groups.'
                )

    else:
        is_logger = True
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        if 'seed' in config.distributed:
            seed = config.distributed.seed

    #Set device, random seed and optimization
    if torch.cuda.is_available():

        if device.type == "cuda":
            torch.cuda.set_device(device.index)

        if 'seed' in config.distributed:
            torch.cuda.manual_seed(seed)
        increase_l2_fetch_granularity()
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass
        
        torch.backends.cudnn.benchmark = True

    if 'seed' in config.distributed:
        torch.manual_seed(seed)

    # Ensure all ranks see the same setup completion before data loading.
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available() and device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()

    return device, is_logger


def increase_l2_fetch_granularity():
    try:
        import ctypes

        _libcudart = ctypes.CDLL('libcudart.so')
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    except:
        return
