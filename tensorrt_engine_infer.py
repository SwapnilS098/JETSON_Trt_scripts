

"""
    Does not requrie any special environment

    Script generates the Engine models from the ONNX version of the Deep Learning model
    If the Engine version of the model does not exists then the Engine file is generated and
    exported to the disc

    Inference from the engine version of the deep learning model is developed.
    
"""



import tensorrt as trt
print(trt.__version__)
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
#import torch
from PIL import Image
import time
import cv2
import os

class Build_engine:

    def __init__(self,onnx_path,engine_path,input_shape):
        self.onnx_path=onnx_path
        self.engine_path=engine_path
        self.height=input_shape[1]
        self.width=input_shape[2]

    def save_engine(self,engine):
        #serialize the engine
        serialized_engine=engine.serialize()

        #save the serialized engine to the file
        with open(self.engine_path,"wb") as f:
            f.write(serialized_engine)
        print("Engine is saved to the disc")

    def load_engine(self):
        logger=trt.Logger(trt.Logger.WARNING)

        #Read the serialized engine from the file
        with open(self.engine_path,"rb") as f:
            serialized_engine=f.read()
        #Deserialize the engine
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def build_engine(self):
        print("Building engine")
        print("Setting configurations for the engine")
        logger=trt.Logger(trt.Logger.WARNING) #Warning level for logging messages
        builder=trt.Builder(logger)           #object for engine  building process
        #network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) #allows to specify the batch size of 1 during inference
        network=builder.create_network(1) # for keeping the batch size of 1
        parser=trt.OnnxParser(network,logger) #parser object to parse the ONNX model to TensorRT network

        #create a builder configuration
        config=builder.create_builder_config() #object to hold the configuration settings of the engine

        #Set memory pool limit for the workspace
        print("Memory_pool_limit for optimization:",1,"GB")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30) 
        #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30) # 1 GB 31 is 2GB
        #Memory used for temporary data during optimization part
        #Reducing it may speed up the engine building process but may harm the optimization process
        #if it requires more memory
        
        #optimization options
        #----------------------------------------------------------------------------------
        half=False
        int8=False
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        elif int8:
            config.set_flag(trt.BuilderFlag.INT8)

        #To ensure the model runs in FP32 precision
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.INT8)

        #DLA Deep Learning Accelerator disable
        config.clear_flag(trt.BuilderFlag.GPU_FALLBACK)
        

        #strip weights: create and optimize engine without unncessary weights
        #strip_weights=False
        #if strip_weights:
        #    config.set_flag(trt.BuilderFlag.STRIP_PLAN)
        #to remove strip plan from config
        #config.flags&=~(1<<int(trt.BuilderFlag.STRIP_PLAN))

        config.clear_flag(trt.BuilderFlag.TF32)

        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        #Parsing the onnx model to the parser object
        with open(self.onnx_path,"rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print("Parsing the ONNX model done")
        
        #Debugging: Print network inputs and outputs
        print("Number of network inputs:",network.num_inputs)
        print("Number of network Ouputs:",network.num_outputs)

        if network.num_inputs==0:
            print("No inputs found in the network")
            return None

        #set optimization profiles if needed
        input_tensor=network.get_input(0)
        if input_tensor is None:
            print("Error: Input tensor is None")
            return None

        print("Input tensor name:",input_tensor.name)
        profile=builder.create_optimization_profile()
        print("profile created")
        #for each tuple the height and width can be different
        #in this case we have kept them same
        profile.set_shape(input_tensor.name,(1,3,self.height,self.width),
                          (1,3,self.height,self.width),
                          (1,3,self.height,self.width))
        
        print("profile shape set")
        #set_shape sets the shape of the input tensor. Batch,heigh,width. 3 times is because is gives the
        #minimum, optimal and maximum values for the engine to optimize the inputs to the engine version of the model
        config.add_optimization_profile(profile) #adds the optimization profile to the builder configuration
        print("configs added")

        #Build the engine
        serialized_engine=builder.build_serialized_network(network,config)
        if serialized_engine is None:
            print("Failed to build the serialized network")
            return None

        #Deserialize the engine
        print("Building Engine")
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(serialized_engine)
        

        return engine
    

    
class TensorRTInference:
    def __init__(self, engine_path,image_path,output_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.image_path=image_path
        self.output_path=output_path

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

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        #input_data = input_data.astype(np.float32)  # Ensure the data type matches
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
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        #output_tensor = self.outputs[0].host.reshape(input_data.shape)

        #return output_tensor
        return self.outputs[0].host



    def preprocess_image(self,shape):
        height=shape[1]
        width=shape[2]
        print("H:",height,"W:",width)
        
        image=cv2.imread(self.image_path)
        image=cv2.resize(image,(width,height))
        image=image.astype(np.float32)/255.0
        image=image.transpose(2,0,1)
        image=np.expand_dims(image,axis=0)
        return image.ravel()

    def preprocess_gray(self,shape):
        height=shape[1]
        width=shape[2]
        print("H:",height,"W:",width)
        #open the image as grayscale
        image=Image.open(self.image_path).convert("L")

        #resize the image
        image=image.resize((width,height))
        #convert the image to numpy and normalize
        image=np.array(image)/255.0
        #create an empty numpy array
        img_blank=np.zeros((3,height,width))

        img_blank[0]=image

        img_final=np.expand_dims(img_blank,axis=0).astype(np.float32)

        return img_final
        
    
    def postprocess_and_save(self,output,output_shape,gray):
        height=output_shape[1]
        width=output_shape[2]
        print("H:",height,"W:",width)
        output=output.reshape(3,height,width)
        output=output.transpose(1,2,0) #convert to the HWC format
        output=(output*255.0).astype(np.uint8)
        
        if gray==False:
            cv2.imwrite(self.output_path,output)
        else:
            output=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
            cv2.imwrite(self.output_path,output,params=[cv2.IMWRITE_JPEG_QUALITY, 20])


def image_handling(dataset_path):
    lst=os.listdir(dataset_path)
    images=[]
    for image in lst:
        if image.lower().endswith("jpg") or image.lower().endswith("jpeg") or image.lower().endswith("png") or image.lower().endswith("webp"):
            images.append(image)
    print("dataset has:",len(images),"images")
    return images

def main(onnx_model_name,onnx_path_base,engine_path_base,save_path_base,image_path,\
         input_shape,output_shape,engine_name,dataset_path):
    """
    Function to run the whole script
    """
    onnx_path=os.path.join(onnx_path_base,onnx_model_name+".onnx")

    if engine_name=="same":
        engine_path=os.path.join(engine_path_base,onnx_model_name+".engine")
        output_img_name="output_"+onnx_model_name
    else:
        engine_name=onnx_model_name+"_"+engine_name
        engine_path=os.path.join(engine_path_base,engine_name+".engine")
        output_img_name="output_"+engine_name
        
    print("Engine_path:",engine_path)
    #output_img_name="output_"+engine_name
    save_path=os.path.join(save_path_base,output_img_name+".jpg")

    ##########################################
    ####ENGINE BUILDING OR LOADING############
    start=time.time()
    Engine=Build_engine(onnx_path,engine_path,input_shape) #Engine is the object of the Build_engine class

    #check if the engine exists on the path
    if os.path.exists(engine_path):
        engine=Engine.load_engine()
        print("Engine is loaded from the disc")
    else:
        print("Engine not found at the path, Building the engine")
        start=time.time()
        engine=Engine.build_engine()
        end=time.time()
        print("engine is built, Time:",round(end-start,2),"seconds")
        Engine.save_engine(engine)
        print("Engine is exported to the disc")
    print("====================Engine done=============================")
    end=time.time()
    print("Engine time:",round(end-start,2),"seconds")
    ##########################################

    ##########################################
    ###Handling the images for the inference on the whole dataset
    images=image_handling(dataset_path)

    trt_inference=TensorRTInference(engine_path,image_path,save_path)
    
    infer_time_lst=[]
    
    for image in images:
        
        image=trt_inference.preprocess_image(input_shape)
        print("Preprocessing is done for:",image)

        #run inference
        start=time.time()
        output_data=trt_inference.infer(image)
        end=time.time()
        
        infer_time=round(end-start,2)
        print("TRT inference time:",infer_time,"s")
        infer_time_lst.append(infer_time)
        print("====================INFERENCE done=============================")

        start=time.time()
        gray=True
        trt_inference.postprocess_and_save(output_data,output_shape,gray)
        print("====================EXPORTING done=============================")
        end=time.time()
        print("Exporting time:",round(end-start,2),"seconds")
        


    

    

    
    
    
    

    
if __name__=="__main__":

    onnx_model_name="bmshj_halfUHD_ssim_8"
    engine_name="same"#"same" #if no change in the engine file name then leave this else give name
    
    #paths
    engine_path_base=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\tensorrt_scripts\engine_models"#\bmshj_model.engine"
    save_path_base=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS"#\trt_infer_image.png"
    onnx_path_base=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\onnx_scripts\onnx_export\onnx_models"#\bmshj_model.onnx"
    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\FLOPS\image_2.png"

    
    #input_shape=[3,1232,1640] #input shape of the tensor which is expected by the ONNX model
    input_shape=[3,1232,1640]
    #input_shape=[1,720,1280]
    #output_shape = np.array([3, 1232, 1648])  # Desired shape 1648 because the model adds the 8 pixels at the end
    output_shape=[3,1232,1648]
    #output_shape=[1,720,1280]
    #output_shape=[3,1232,1648]
    #running the main function
    start=time.time()
    main(onnx_model_name,onnx_path_base,engine_path_base,save_path_base,image_path,\
         input_shape,output_shape,engine_name,dataset_path)
    end=time.time()
    print("Total Script time:",round(end-start,2),"seconds")
    
    

    ###Learning
    #If the input_shape is not correct which is expected by the onnx model then also the engine building process fails

    

    
    
