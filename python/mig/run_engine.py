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
engine_file_name = "../../engines/" + "build_engine." + "trt"


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