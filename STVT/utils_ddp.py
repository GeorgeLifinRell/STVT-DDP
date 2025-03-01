# import os
# import torch
# import torch.distributed as dist

# def setup(rank: int, world_size: int):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     torch.cuda.set_device(rank)
#     dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

# if __name__ == "__main__":
#     setup(0, 1)
#     print("Hello World")
#     cleanup()

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Process with rank {rank} initialized (out of {world_size} processes)")

def cleanup():
    dist.destroy_process_group()

def demo_fn(rank, world_size):
    # Setup the process group
    setup(rank, world_size)
    
    # Get information about the device this process is using
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}")
    
    # Create a tensor on this device
    tensor = torch.randn(2, 3).to(device)
    print(f"Process {rank} tensor: {tensor}")
    
    # Clean up
    cleanup()
    print(f"Process {rank} completed")

if __name__ == "__main__":
    # Define number of processes to spawn
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} CUDA devices")
    
    if world_size > 1:
        print(f"Launching {world_size} processes...")
        mp.spawn(
            demo_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        print("Running with a single process")
        demo_fn(0, 1)
    
    print("All processes finished")