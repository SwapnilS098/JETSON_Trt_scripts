import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import io
import asyncio

class InferenceAsync:
    def __init__(self, engine_path, input_shape, output_shape):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.image_queue = asyncio.Queue()

        # Allocate Buffers
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

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropriate input/output lists
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    async def infer(self, input_data):
        input_data = input_data.astype(np.float32)
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[1].host, self.outputs[1].device, self.stream)
        self.stream.synchronize()
        return self.outputs[1].host

    async def preprocess_gray_2(self, image_buffer):
        height, width = self.input_shape[1], self.input_shape[2]
        image = Image.open(image_buffer).resize((width, height)).convert("L")
        image = np.array(image) / 255.0
        img_final = np.zeros((1, 3, height, width), dtype=np.float32)
        img_final[0, 0, :, :] = image
        return img_final.ravel()

    async def postprocess_gray_final(self, output, quality):
        height, width = self.output_shape[1], self.output_shape[2]
        output = (np.clip(output, 0, 1)*255.0).astype(np.uint8).reshape(3,height,width)
        
        image_data = output[0]

        #image = Image.fromarray(image, mode='L')
        #buffer = io.BytesIO()
        #image.save(buffer, format="AVIF", quality=quality)
        
        #running the image encoding in separater thread
        loop=asyncio.get_running_loop()
        buffer=await loop.run_in_executor(None,self._encode_to_buffer,image_data,quality)


        return buffer

    def _encode_to_buffer(self,image_data,quality):
        """helper function to save the image to AVIF buffer in the separate thread"""
        image=Image.fromarray(image_data,mode="L")
        buffer=io.BytesIO()
        image.save(buffer,format="JPEG",quality=quality)
        return buffer


    async def handle_image(self, image_buffer):
        preprocessed_image = await self.preprocess_gray_2(image_buffer)
        inference_output = await self.infer(preprocessed_image)
        processed_image_buffer = await self.postprocess_gray_final(inference_output, quality=20)
        return processed_image_buffer

    async def listen_for_images(self):
        while True:
            image_buffer = await self.image_queue.get()
            processed_image_buffer = await self.handle_image(image_buffer)
            # Do something with the processed image buffer, like saving or sending it elsewhere
            print("Image processed and ready for next steps.")

    async def enqueue_image(self, image_buffer):
        await self.image_queue.put(image_buffer)
        print("Image added to the queue.")

async def main():
    engine_path = 'your_engine.trt'
    input_shape = (1, 3, 224, 224)  # Example input shape
    output_shape = (1, 3, 224, 224)  # Example output shape

    inference_system = InferenceAsync(engine_path, input_shape, output_shape)
    asyncio.create_task(inference_system.listen_for_images())

    # Example of adding an image to the queue (normally this would come from an external source)
    sample_image_buffer = io.BytesIO()  # Replace with actual image buffer
    await inference_system.enqueue_image(sample_image_buffer)

if __name__ == "__main__":
    asyncio.run(main())

