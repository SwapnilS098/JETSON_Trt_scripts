import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import io
import time

class Inference:
    def __init__(self, engine_path, input_shape, output_shape, image_buffer):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.image_buffer = image_buffer

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to bindings
            bindings.append(int(device_mem))

            # Append to inputs/outputs list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        input_data = input_data.astype(np.float32)
        expected_shape = self.inputs[0].host.shape
        print(f"Input data shape: {input_data.shape}, Expected shape: {expected_shape}")

        if input_data.size != np.prod(expected_shape):
            raise ValueError(f"Input data size mismatch: {input_data.size} vs {np.prod(expected_shape)}")

        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()

        return self.outputs[0].host

    def preprocess_gray_2(self, image_buffer):
        height, width = self.input_shape[1], self.input_shape[2]
        image_size_kb = image_buffer.getbuffer().nbytes / 1024

        image = Image.open(image_buffer).resize((width, height)).convert("L")
        image = np.array(image) / 255.0

        img_final = np.zeros((1, 3, height, width), dtype=np.float32)
        img_final[0, 0, :, :] = image

        return img_final.ravel(), image_size_kb

    def postprocess_gray_final(self, output, quality):
        height, width = self.output_shape[1], self.output_shape[2]
        output = np.clip(output, 0, 1)
        output = (output * 255.0).astype(np.uint8).reshape(3, height, width)
        
        image = Image.fromarray(output[0], mode='L')

        buffer = io.BytesIO()
        image.save(buffer, format="AVIF", quality=quality)
        buffer_size = len(buffer.getvalue()) / 1024

        print("Exported image size in buffer form is:", buffer_size, "KB")
        return buffer_size, buffer

    def main(self):
        desired_size = 200  # in KB
        quality = 20

        image_, image_size = self.preprocess_gray_2(self.image_buffer)
        print("Image size:", image_size, "KB")

        start = time.time()
        output_data = self.infer(image_)
        end = time.time()
        
        img_size, buffer_image = self.postprocess_gray_final(output_data, quality)
        print("Exported image size:", img_size, "KB")

        return buffer_image

