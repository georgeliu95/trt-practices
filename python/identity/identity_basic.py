import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import sys
sys.path.insert(0, '..')
from common import HostDeviceMem, BuilderCreationFlag
from common import allocate_buffer, printIOInfo, get_trt_type


# -------------------------------------------------------------------------------- #
# Global Variables
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
WORKSPACE_SIZE = 1<<30
BATCH_MODE = 1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) if True else 0
builder_flag = BuilderCreationFlag()
output_tensor_names = []
engine_file_name = __file__[:-2] + "trt"


def trt_execute(context, input_data):
    # Execution
    context.set_binding_shape(0, input_data.shape)
    inputs, outputs, bindings, stream = allocate_buffer(context)

    # Copy data to Host Mem
    [np.copyto(inputs[0].host, input_data.ravel().astype(inputs[0].host.dtype))]
    # Copy data to Device Mem
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return inputs, outputs, bindings, stream


# -------------------------------------------------------------------------------- #
# Data Preparation
input_dtype = np.float32
input_shape = [3,2,2,2]
input_data = np.arange(start=0,stop=np.prod(input_shape),dtype=input_dtype).reshape(input_shape)
print("Input:\n", input_data if input_data.size < 50 else input_data[:][:][:min(5,input_shape[-2])][:min(5,input_shape[-1])])


# -------------------------------------------------------------------------------- #
# Network Definition
builder = trt.Builder(TRT_LOGGER)
builder.max_workspace_size = WORKSPACE_SIZE

network = builder.create_network(BATCH_MODE)
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = builder.max_workspace_size
config.set_flag(builder_flag.ld("tf32"))

input_tensor = network.add_input(name="input_tensor_0", dtype=get_trt_type(input_dtype), shape=(-1,-1,-1,-1))
profile.set_shape("input_tensor_0", min=(1,1,1,1),
                                  opt=(100,100,100,100),
                                  max=(100,100,100,100))
config.add_optimization_profile(profile)


# ----- MODIFY CODE HERE ----- #
identity_layer = network.add_identity(input_tensor)
identity_layer.name = "identity_layer"
# identity_layer.get_output(0).dtype = trt.float32


# ----- MODIFY CODE HERE ----- #
output_layer = [identity_layer]
output_tensor = [it.get_output(0) for it in output_layer]
for i,it in enumerate(output_tensor):
    it.name = "output_tensor_" + str(i)
    output_tensor_names.append(it.name)
[network.mark_output(it) for it in output_tensor]


# CUDA Engine
engine = builder.build_engine(network, config)
# Execution Context
context = engine.create_execution_context()
# Destroy Network Definition and Builder
with network, builder:
    pass

_, outputs, _, _ = trt_execute(context, input_data)

printIOInfo(engine, context)
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
_, outputs, _, _ = trt_execute(des_context, input_data)

[print("Output_"+str(id)+":\n", outputs[id].host.reshape(des_context.get_binding_shape(des_engine.get_binding_index(it)))) for id,it in enumerate(output_tensor_names)]


# Destroy Execution Context and CUDA Engine
with context:
    print("destroy context")
with engine:
    print("destroy engine")

with des_context:
    print("destroy context")
with des_engine:
    print("destroy engine")
print("All done.")
