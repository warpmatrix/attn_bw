import torch
import math


def clone_tensor(tensor_shape, dtype):
    src = torch.rand(tensor_shape, dtype=dtype, device="cuda")
    dst = src.clone()
    return dst


if __name__ == "__main__":
    num_warmups, num_execs = 3, 3
    tensor_shape = (1024, 1024, 1024, 8)
    dtype = torch.float16
    for _ in range(num_warmups):
        clone_tensor(tensor_shape, dtype)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_execs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_execs)]
    for i in range(num_execs):
        starts[i].record()
        clone_tensor(tensor_shape, dtype)
        ends[i].record()
    torch.cuda.synchronize()
    elapsed_times = [starts[i].elapsed_time(ends[i]) for i in range(num_execs)]  # ms
    size = math.prod(tensor_shape) * dtype.itemsize / (1024**2)  # MB
    bandwidths = [size / time for time in elapsed_times]  # GB/s
    print(bandwidths)
