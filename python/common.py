import tensorrt as trt
import pycuda.driver as cuda
import numpy as np


# # A way to init cuda without pycuda.autoinit
# # CUDA Context Init
# cuda.init()
# CURRENT_DEV = cuda.Device(0)
# ctx = CURRENT_DEV.make_context()
# ctx.push()
# # ... Your code here ...
# # Pop CUDA Context
# ctx.pop()
# ctx.detach()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()


class BuilderCreationFlag():
    def __init__(self):
        self.__trt_flags_dict = {"half":trt.BuilderFlag.FP16,
                                "tf32":trt.BuilderFlag.TF32,
                                "int8":trt.BuilderFlag.INT8,
                                "debug":trt.BuilderFlag.DEBUG,
                                "strict_type":trt.BuilderFlag.STRICT_TYPES,
                                "refit":trt.BuilderFlag.REFIT,
                                "share_time_cache":trt.BuilderFlag.DISABLE_TIMING_CACHE}

    def ld(self, flag_name:str):
        return self.__trt_flags_dict[flag_name]


def get_trt_type(np_type:np.dtype)->trt.DataType:
    if np_type == np.float32:
        return trt.float32
    elif np_type == np.float16:
        return trt.float32
    elif np_type == np.int32:
        return trt.int32
    elif np_type == np.int8:
        return trt.int8
    else:
        raise ValueError("{} is not supported.".format(str(np_type)))


def is_dynamic_shape(context)->bool:
    engine = context.engine
    if engine.has_implicit_batch_dimension:
        return False
    for idx in range(engine.num_bindings):
        if engine.binding_is_input(idx) and (-1 in engine.get_binding_shape(idx)):
            return True
    return False


def allocate_buffer(context, profile_id:int=0, stream=None)->tuple:
    engine = context.engine
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream() if stream is None else stream
    num_profiles = engine.num_optimization_profiles
    assert(engine.num_bindings % num_profiles ==0)
    num_bindings = engine.num_bindings // num_profiles
    # Attain size with context if tensor is dynamic shape
    bindle = context if is_dynamic_shape(context) else engine
    for binding in range(profile_id*num_bindings, (profile_id+1)*num_bindings):
        size = trt.volume(bindle.get_binding_shape(binding)) * engine.max_batch_size # engine.max_batch_size is always 1 in explicit batch mode
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


def printIOInfo(engine, context=None)->None:
    dynamic_shape = is_dynamic_shape(context) if context is not None else False
    for i in range(engine.num_bindings):
        print("input" if engine.binding_is_input(i) else "output", 
                engine.get_binding_name(i), 
                engine.get_binding_dtype(i), 
                engine.get_binding_shape(i), context.get_binding_shape(i) if dynamic_shape else None)


