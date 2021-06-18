import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
# import nvtx

import os
import sys
sys.path.insert(0, '..')
from common import HostDeviceMem, BuilderCreationFlag
from common import allocate_buffer, printIOInfo, get_trt_type

# CUDA Context Init
cuda.init()
CURRENT_DEV = cuda.Device(0)
ctx = CURRENT_DEV.make_context()
ctx.push()


# -------------------------------------------------------------------------------- #
# Global Variables
BENCH = False
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
WORKSPACE_SIZE = 1<<30
BATCH_MODE = 1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) if True else 0
builder_flag = BuilderCreationFlag()
output_tensor_names = []
engine_file_name = "../../engines/" + __file__[:-2] + "trt"


def trt_execute(context, input_data):
    # Execution
    inputs, outputs, bindings, stream = allocate_buffer(context)

    # Copy data to Host Mem
    [np.copyto(inputs[0].host, input_data.ravel().astype(inputs[0].host.dtype))]
    # Copy data to Device Mem
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    if BENCH:
        for i in range(100):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return inputs, outputs, bindings, stream


# -------------------------------------------------------------------------------- #
# Data Preparation
input_dtype = np.float32
input_shape = [3,2,5,5]
input_data = np.arange(start=0,stop=np.prod(input_shape),dtype=input_dtype).reshape(input_shape)
print("Input:\n", input_data if input_data.size < 50 else input_data[:][:][:min(5,input_shape[-2])][:min(5,input_shape[-1])])


# -------------------------------------------------------------------------------- #
# Network Definition
builder = trt.Builder(TRT_LOGGER)
#builder.max_workspace_size = WORKSPACE_SIZE

network = builder.create_network(BATCH_MODE)
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = WORKSPACE_SIZE
config = builder_flag.set(config, ["tf32"])

input_tensor = network.add_input(name="input_tensor_0", dtype=get_trt_type(input_dtype), shape=(-1,2,-1,-1))
profile.set_shape("input_tensor_0", min=(1,2,1,1),
                                  opt=(100,2,100,100),
                                  max=(100,2,100,100))
config.add_optimization_profile(profile)


# ----- MODIFY CODE FOR NETWORK HERE ----- #
identity_layer = network.add_identity(input_tensor)
identity_layer.name = "identity_layer_1"
# identity_layer.get_output(0).dtype = trt.float32

conv_layer = network.add_convolution_nd(identity_layer.get_output(0), num_output_maps=20, kernel_shape=(3,3), 
                                        kernel=np.random.randn(20,2,3,3).astype(np.float32))
conv_layer.name = "conv_layer_1"

out_layer = network.add_identity(conv_layer.get_output(0))
out_layer.name = "out_layer_1"


# ----- MODIFY CODE FOR OUTPUT HERE ----- #
output_layer = [out_layer]
output_tensor = [it.get_output(0) for it in output_layer]
for i,it in enumerate(output_tensor):
    it.name = "output_tensor_" + str(i)
    output_tensor_names.append(it.name)
[network.mark_output(it) for it in output_tensor]


# CUDA Engine
engine = builder.build_engine(network, config)
# Execution Context
context = engine.create_execution_context()
context.set_binding_shape(0, input_data.shape) # Only one input tensor here.
# Destroy Network Definition and Builder
with network, builder:
    pass
printIOInfo(engine, context)

# rng = nvtx.start_range(message="execution_phase", color="blue")
_, outputs, _, _ = trt_execute(context, input_data)
# nvtx.end_range(rng)

[print("Output_"+str(id)+":\n", outputs[id].host.reshape(context.get_binding_shape(engine.get_binding_index(it)))) for id,it in enumerate(output_tensor_names)]


# Serialization and deserialization test
try:
    os.remove(engine_file_name)
except FileNotFoundError:
    print("[Remove engine] {} is not found.".format(engine_file_name))
with open(engine_file_name, "wb") as f:
    f.write(engine.serialize())

if os.path.isfile(engine_file_name):
    with open(engine_file_name, "rb") as f:
        engine_bs = f.read()
    des_engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_bs)
    des_msg = "[Deserialize engine] " + ("Fail to load engine." if des_engine is None else "Succeed to load engine.")
    print(des_msg)

des_context = des_engine.create_execution_context()
des_context.set_binding_shape(0, input_data.shape) # Only one input tensor here.

# rng = nvtx.start_range(message="execution_phase", color="blue")
_, outputs, _, _ = trt_execute(des_context, input_data)
# nvtx.end_range(rng)

[print("Output_"+str(id)+":\n", outputs[id].host.reshape(des_context.get_binding_shape(des_engine.get_binding_index(it)))) for id,it in enumerate(output_tensor_names)]

# Destroy Execution Context and CUDA Engine
#with context:
#    print("destroy context")
#with engine:
#    print("destroy engine")

#with des_context:
#    print("destroy des_context")
#with des_engine:
#    print("destroy des_engine")
print("All done.")

# Pop CUDA Context
ctx.pop()
ctx.detach()