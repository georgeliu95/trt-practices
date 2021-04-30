import numpy as np
import tensorrt as trt
import nvtx
import pycuda.driver as cuda
import pycuda.autoinit

import os
import sys
sys.path.insert(0, '..')
from common import HostDeviceMem, BuilderCreationFlag
from common import allocate_buffer, printIOInfo, get_trt_type


# -------------------------------------------------------------------------------- #
# Global Variables
BENCH = True
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
channel = 2
input_shape = [3,channel,6,5]
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
config = builder_flag.set(config, ["tf32", "fp16", "strict_type"])

input_tensor = network.add_input(name="input_tensor_0", dtype=get_trt_type(input_dtype), shape=(-1,channel,-1,-1))
profile.set_shape("input_tensor_0", min=(1,channel,1,1),
                                  opt=(100,channel,100,100),
                                  max=(100,channel,100,100))
config.add_optimization_profile(profile)


# ----- MODIFY CODE FOR NETWORK HERE ----- #
preprocess_layer = network.add_scale(input_tensor,
                                    mode=trt.ScaleMode.UNIFORM,
                                    scale=None,
                                    shift=np.array([1] ,dtype=np.float32),
                                    power=None)
preprocess_layer.name = "preprocess_layer"


conv_layer_1 = network.add_convolution(preprocess_layer.get_output(0),
                                        num_output_maps=2,
                                        kernel_shape=[3,3],
                                        kernel=np.ones((2,2,3,3),dtype=np.float32),
                                        bias=None)
conv_layer_1.name = "conv_layer_1"
out_layer_1 = network.add_activation(conv_layer_1.get_output(0), type=trt.ActivationType.RELU)
out_layer_1.name = "out_layer_1"


conv_layer_2 = network.add_convolution(preprocess_layer.get_output(0),
                                        num_output_maps=2,
                                        kernel_shape=[3,3],
                                        kernel=np.ones((2,2,3,3),dtype=np.float32),
                                        bias=None)
conv_layer_2.name = "conv_layer_2"
identity_layer_1 = network.add_identity(conv_layer_2.get_output(0))
identity_layer_1.name = "identity_layer_1"
out_layer_2 = network.add_activation(identity_layer_1.get_output(0), type=trt.ActivationType.RELU)
out_layer_2.name = "out_layer_2"


elementadd_layer_1 = network.add_elementwise(out_layer_2.get_output(0), out_layer_2.get_output(0), op=trt.ElementWiseOperation.SUM)
elementadd_layer_1.name = "elementadd_layer_1"
out_layer_3 = network.add_activation(elementadd_layer_1.get_output(0), type=trt.ActivationType.RELU)
out_layer_3.name = "out_layer_3"


elementadd_layer_2 = network.add_elementwise(out_layer_1.get_output(0), out_layer_1.get_output(0), op=trt.ElementWiseOperation.SUM)
elementadd_layer_2.name = "elementadd_layer_2"
identity_layer_2 = network.add_identity(elementadd_layer_2.get_output(0))
identity_layer_2.name = "identity_layer_2"
out_layer_4 = network.add_activation(identity_layer_2.get_output(0), type=trt.ActivationType.RELU)
out_layer_4.name = "out_layer_4"

# ----- MODIFY CODE FOR OUTPUT HERE ----- #
output_layer = [out_layer_1, out_layer_2, out_layer_3, out_layer_4]
output_tensor = [it.get_output(0) for it in output_layer]
for i,it in enumerate(output_tensor):
    it.name = "output_tensor_" + str(i)
    output_tensor_names.append(it.name)
[network.mark_output(it) for it in output_tensor]
network.get_output(0).dtype = trt.float16


# CUDA Engine
engine = builder.build_engine(network, config)
# Execution Context
context = engine.create_execution_context()
context.set_binding_shape(0, input_data.shape) # Only one input tensor here.
# Destroy Network Definition and Builder
with network, builder:
    pass
printIOInfo(engine, context)


rng = nvtx.start_range(message="execution_phase", color="blue")
_, outputs, _, _ = trt_execute(context, input_data)
nvtx.end_range(rng)

[print("Output_"+str(id)+":\n", outputs[id].host.reshape(context.get_binding_shape(engine.get_binding_index(it)))) for id,it in enumerate(output_tensor_names)]


# Destroy Execution Context and CUDA Engine
with context:
    print("destroy context")
with engine:
    print("destroy engine")

print("All done.")
