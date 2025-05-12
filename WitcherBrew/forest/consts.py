"""Setup constants, ymmv."""

PIN_MEMORY = True
NON_BLOCKING = True
BENCHMARK = True
MAX_THREADING = 40
SHARING_STRATEGY = 'file_descriptor'  # file_system or file_descriptor 
# file_system strategy can be useful when sharing tensors between processes running on different machines or when the tensor data is too large to fit in shared memory.

DEBUG_TRAINING = False

DISTRIBUTED_BACKEND = 'gloo'  # nccl would be faster, but require gpu-transfers for indexing and stuff

