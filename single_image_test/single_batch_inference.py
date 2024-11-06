
"""
    developing a new script which will take the input as the Python buffer object and then 
    it will be fed to the DL model for the 
    1.preprocessing
    2. inference
    3.post processing and exporting the batch as the Python buffer object

"""

#load the Engine model to the disc
# preprocess the image 
# run the inference 
# post process the image 

import tensorrt as trt
print(trt.__version__)
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import time
import os
import pillow_avif
import io


class Inference:
    def __init__(self,engine_path,input_shape,output_shape,image_buffer):
        self.logger=trt.Logger(trt.Logger.ERROR)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.load_engine(engine_path)
        self.context=self.engine.create_execution_context()
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.image_buffer=image_buffer

        #allocate Buffers
        self.inputs,self.outputs,self.bindings,self.stream=self.allocate_buffers(self.engine)

    def load_engine(self,engine_path):
        with open(engine_path,"rb") as f:
            engine=self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self,host_mem,device_mem):
            self.host=host_mem
            self.device=device_mem

    def allocate_buffers(self,engine):
        inputs,outputs,bindings=[],[],[]
        stream=cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name=engine.get_tensor_name(i)
            size=trt.volume(engine.get_tensor_shape(tensor_name))
            dtype=trt.nptype(engine.get_tensor_dtype(tensor_name))

            #Allocate host and device buffers
            host_mem=cuda.pagelocked_empty(size,dtype)
            device_mem=cuda.mem_alloc(host_mem.nbytes)

            #append the device buffer address to device bindings
            bindings.append(int(device_mem))

            #append to the appropriate input/output_lists
            if engine.get_tensor_mode(tensor_name)==trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem,device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem,device_mem))

        return inputs,outputs,bindings,stream

    def infer(self, input_data):
        input_data = input_data.astype(np.float32)  # Ensure the data type matches
        expected_shape = self.inputs[0].host.shape
        print(f"Input data shape: {input_data.shape}, Expected shape: {expected_shape}")

        # Check if the shape of input_data matches expected shape
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
        cuda.memcpy_dtoh_async(self.outputs[1].host, self.outputs[1].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        #output_tensor = self.outputs[0].host.reshape(input_data.shape)
        #changing the outputs[0] to outputs[1]
        #print("size of the array from the infer function:",self.outputs[1].host.shape)
        #return output_tensor
        return self.outputs[1].host

    

    def preprocess_gray_2(self,image_buffer):
        """
            this function is the modified version of the preprocess_gray_1 where the images are
            read from the buffer object instead of the path"""

        height,width=self.input_shape[1],self.input_shape[2]

        image_size_kb=image_buffer.getbuffer().nbytes/1024 #len(image_buffer)/1024 #get the image size in KB

        image = Image.open(image_buffer).resize((width, height)).convert("L")
        image = np.array(image) / 255.0

        # Stack the grayscale image into the first channel of a blank 3D array
        img_final = np.zeros((1, 3, height, width), dtype=np.float32)
        img_final[0, 0, :, :] = image

        # Flatten and return
        return img_final.ravel(),image_size_kb
    
    def postprocess_gray_final(self,output,quality):
        """
            This is the function for the postprocessing and the saving
            of the image in the gray form which is of the 3 channel.

            The results will be in the 3 channel with the first channel contains the
            data and the rest of them will have be zero

            also this function will export the image to the buffer and then the calculation will be made from it


            """
        print("output shape is:",self.output_shape)
        height=self.output_shape[1]
        width=self.output_shape[2]

        #clip the output
        output=np.clip(output,0,1)

        #denormalization and the datatype to integer type
        output=(output*255.0).astype(np.uint8)

        output=output.reshape(3,height,width)

        image=output[0]


        image=Image.fromarray(image,mode='L')

        #save the image to the memory buffer

        buffer=io.BytesIO()
        image.save(buffer,format="AVIF",quality=quality)

        #get the size of the buffer
        buffer_size=len(buffer.getvalue())/1024 #in KB
        print("exported_image_size in buffer form is:",buffer_size,"KB")

        return buffer_size,buffer

    def main(self):
        
        desired_size=200 # in KB
        quality=20  #this is the AVIF encoder quality parameter
        
        #preprocess the image
        image_,image_size=self.preprocess_gray_2(self.image_buffer)
        print("image size:",image_size,"KB")

        #run inference on the model 
        start=time.time()
        output_data=self.infer(image_) #inference step
        end=time.time()
        
        #export the image 
        img_size,buffer_image=self.postprocess_gray_final(output_data,quality)
        print("Exported image_size is:",img_size,"KB")
        
        return buffer_image


if __name__=="__main__":

    onnx_model_name="bmshj_4_UHD"
    input_shape=[3,2464,3280]
    output_shape=[3,2464,3280]
    engine_path=r"/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts/engine_models/bmshj_4_UHD.engine"
    
    Inference_object=Inference(engine_path,input_shape,output_shape,image_buffer)
    Inference_object.main()
