using CUDA


a = CUDA.DEVICE_ATTRIBUTE_MULTI_GPU_BOARD
@show multigpuboard = CUDA.attribute(device(), a)

a = CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
@show memclockrate = CUDA.attribute(device(), a)

a = CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
@show shmemperblock = CUDA.attribute(device(), a)

a = CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
@show shmempersm = CUDA.attribute(device(), a)

a = CUDA.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
@show regsperblock = CUDA.attribute(device(), a)

a = CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
@show threadspersm = CUDA.attribute(device(), a)

a = CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE
CUDA.attribute(device(), a)

CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
CUDA.attribute(device(), a)

CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
CUDA.attribute(device(), a)

CUDA.CU_DEVICE_ATTRIBUTE

a = 0
CUDA.cuDeviceGetProperties(a, device())


b = CUDA.CUdevprop_st
