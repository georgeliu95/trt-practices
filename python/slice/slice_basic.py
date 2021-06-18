import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()


def get_trt_type(np_type):
    if np_type == np.float32:
        return trt.float32
    elif np_type == np.float16:
        return trt.float32
    elif np_type == np.int32:
        return trt.int32
    elif np_type == np.int8:
        return trt.int8
    else:
        print("{} is not supported.".format(str(np_type)))
        return None


def allocate_buffer(engine, is_dynamic_shape:bool=False, profile_id:int=0, context=None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    num_profiles = engine.num_optimization_profiles
    assert(engine.num_bindings % num_profiles ==0)
    num_bindings = engine.num_bindings // num_profiles
    # Attain size with context if tensor is dynamic shape
    bindle = context if is_dynamic_shape else engine

    for binding in range(profile_id*num_bindings, (profile_id+1)*num_bindings):
        size = trt.volume(bindle.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def printEngineInfo(engine):
    for i in range(engine.num_bindings):
        print("input" if engine.binding_is_input(i) else "output", 
                engine.get_binding_name(i), 
                engine.get_binding_dtype(i), 
                engine.get_binding_shape(i))

# Global Variables
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
WORKSPACE_SIZE = 3<<30
TRT_NETWORK_FLAG = 1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Data Preparation
input_dtype = np.float32
input_shape = [3,2,2,3]
input_data = np.arange(start=0,stop=np.prod(input_shape),dtype=input_dtype).reshape(input_shape)
print("Input:\n", input_data)

# -------------------------------------------------------------------------------- #
# Network Definition
builder = trt.Builder(TRT_LOGGER)
builder.max_workspace_size = WORKSPACE_SIZE

network = builder.create_network(0)
# profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = builder.max_workspace_size
# config.set_flag(trt.BuilderFlag.FP16)

builder.max_batch_size = 5
input_tensor = network.add_input(name="input_tensor", dtype=get_trt_type(input_dtype), shape=input_shape[1:])#(-1,-1,-1,-1))
# profile.set_shape("input_tensor", min=(1,1,1,1),
#                                   opt=(10,10,10,10),
#                                   max=(100,100,100,100))
# config.add_optimization_profile(profile)

preprocess_layer = network.add_identity(input_tensor)
preprocess_layer.name = "preprocess_layer"

# shape_layer = network.add_shape(preprocess_layer.get_output(0))
# print(shape_layer)
# const_layer_1 = network.add_constant(shape=(3,), weights=np.array([1,1,1],dtype=np.int32))
# const_layer_2 = network.add_constant(shape=(3,), weights=np.array([2,2,2],dtype=np.int32))
# new_shape_layer_1 = network.add_elementwise(shape_layer.get_output(0), const_layer_1.get_output(0), op=trt.ElementWiseOperation.SUM)
# new_shape_layer_2 = network.add_elementwise(new_shape_layer_1.get_output(0), const_layer_2.get_output(0), op=trt.ElementWiseOperation.DIV)



slice_layer = network.add_slice(preprocess_layer.get_output(0), start=(0,0,0), shape=[it*2 for it in preprocess_layer.get_output(0).shape], stride=(1,1,0))
# slice_layer.set_input(2, shape_layer.get_output(0))
slice_layer.mode = trt.SliceMode.WRAP

output_layer = [slice_layer]
output_tensor = [it.get_output(0) for it in output_layer]
for i,it in enumerate(output_tensor):
    it.name = "output_" + str(i)
[network.mark_output(it) for it in output_tensor]

# CUDA Engine
engine = builder.build_engine(network, config)
# Execution Context
context = engine.create_execution_context()
print(engine.has_implicit_batch_dimension)

# Execution
# context.set_binding_shape(0, input_data.shape)
# inputs, outputs, bindings, stream = allocate_buffer(engine, True, 0, context)
inputs, outputs, bindings, stream = allocate_buffer(engine, False, 0, context)

# Copy data to Host Mem
[np.copyto(inputs[0].host[:input_data.size], input_data.ravel().astype(inputs[0].host.dtype))]

[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
# test_mem = cuda.pagelocked_empty(inputs[0].device., dtype)

# context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
context.execute_async(batch_size=5, bindings=bindings, stream_handle=stream.handle)
[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
stream.synchronize()

printEngineInfo(engine)
print(outputs[0].host[:10])#.reshape(context.get_binding_shape(engine.get_binding_index("output_0") * builder.max_batch_size)))

print("Input shape=", input_data.shape)
print("Output shape=", context.get_binding_shape(engine.get_binding_index("output_0")))
# Destroy Network Definition and Builder
with network, builder:
    pass
# Destroy Execution Context and CUDA Engine
with context:
    print("destroy context")
with engine:
    print("destroy engine")
print("All done.")
