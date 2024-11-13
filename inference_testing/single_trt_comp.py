"""
Script to run the DL image compression over the images
using TENSORRT.

-Main requirement TensorrT must be installed (Made using Tensorrt=10.3.0)
"""

#Importing the modules
import tensorrt as trt
print("Using:",trt.__version__)

import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import time
import os
import pillow_avif
from tqdm import tqdm


class DL_compression:

    def __init__(self,args):
        self.engine_path = args.engine_path
        
        self.dataset_path=args.dataset_path
        
        if os.path.exists(args.project_path)==False:
            os.mkdir(args.project_path)
        self.project_path=args.project_path
        
        #input shape is fixed by the Engine file
        #hence hardcoded here
        self.height=args.input_shape[0]
        self.width=args.input_shape[1]
        
        self.input_shape=(3,self.height,self.width)
        #self.input_shape=(3,1232,1640)
        
        
        #use the original image size only
        if self.input_shape[1]==1232 and self.input_shape[2]==1640:
            self.output_shape=(3,1232,1648)
        elif self.input_shape[1]==540 and self.input_shape[2]==960:
            self.output_shape=(3,544,960)
        else:
            self.output_shape=self.input_shape
        
        print("input_shape:",self.input_shape)
        print("output_shape:",self.output_shape)

        self.quality=60
        self.img_format="AVIF" #"JPEG"
        
        
    
    def load_engine(self):
        
        """Function to load the engine using the engine path
            and return the engine object.
        """
        
        #loading the engine
        logger = trt.Logger(trt.Logger.WARNING)
        #read the serialized engine
        with open(self.engine_path,"rb") as f:
            serialized_engine=f.read()
            
        #deserialize the engine
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(serialized_engine)
        return engine
    
    
    
    
    
    def main(self):
        
        #first load the engine
        engine=self.load_engine()
        
        #arguments for the Tensorrt 
        dataset_path=self.dataset_path
        export_dataset_path=self.project_path
        input_shape=self.input_shape
        output_shape=self.output_shape
        
        #get the inference class
        inference=TensorRTInference(engine,dataset_path,export_dataset_path,input_shape,output_shape,self.quality,self.img_format)
        inference.inference_over_dataset()
        
        
        
    

class TensorRTInference:
    def __init__(self, engine,dataset_path,export_data_path,input_shape,output_shape,quality,img_format):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = engine  #self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.dataset_path=dataset_path
        self.export_data_path=export_data_path
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.gray=False #Ignored for now. Later it can be used for making the code switchable between RGB and gray
        self.quality=quality
        self.img_format=img_format#"AVIF"#"JPEG"
        self.input_image_mode=None
        

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
        #print(f"Input data shape: {input_data.shape}, Expected shape: {expected_shape}")

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
        #cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        cuda.memcpy_dtoh_async(self.outputs[1].host,self.outputs[1].device,self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        #output_tensor = self.outputs[0].host.reshape(input_data.shape)

        #return output_tensor
        return self.outputs[1].host
        #return self.outputs[0].host

    def image_handling(self):
        lst=os.listdir(self.dataset_path)
        images=[]
        for image in lst:
            if image.lower().endswith("jpg") or image.lower().endswith("jpeg") or image.lower().endswith("png") or image.lower().endswith("webp") or image.lower().endswith("pgm"):
                images.append(image)
        print("dataset has:",len(images),"images")
        
        #write code to get the information about the image resolution and the image format and image mode
        
        #information about the image resolution
        #open the image using the PIL 
        image_path=os.path.join(self.dataset_path,images[0])
        image=Image.open(image_path) #open the image
        width=image.size[1]
        height=image.size[0]
        mode=image.mode
        format=images[0].split('.')[-1]
        
        print("Image resolution is:",width,"x",height)
        print("Image mode is:",mode) #mode will be RGB or L
        print("Image format is:",format)      
        
        #reset the input image  mode
        self.input_image_mode=mode

        return images

    def inference_over_dataset(self):
        """
        This method runs the inference over the whole dataset
        and exports the output to the disc
        """
        print("Running the inference_over_dataset")
        images=self.image_handling()

        infer_time_lst=[]
        preprocess_time_lst=[]
        overall_time_lst=[]
        print("Using:",self.img_format,"using quality:",self.quality)
        
        if self.img_format=="JPEG":
                extension=".jpg"
        elif self.img_format=="AVIF":
            extension=".avif"
        elif self.img_format=="WEBP":
            extension=".webp"
        elif self.img_format=="PNG":
            extension=".png"
        else:
            print("PROBLEM in image format, need to check TensorRTInference class")
            return 
        
        if self.input_image_mode=="RGB":
        
            for image in tqdm(images):

                start_=time.time()
                
                start=time.time()
                image_path=os.path.join(self.dataset_path,image)
                #image_=self.preprocess_image(image_path)
                image_=self.preprocess_gray_new_jetson(image_path)
                end=time.time()
                preprocess_time=round(end-start,2)
                #print("preprocess is done for:",image,"time:",preprocess_time,"seconds")
                preprocess_time_lst.append(preprocess_time)

                #run inference
                start=time.time()
                output_data=self.infer(image_) #inference step
                end=time.time()
                
                infer_time=round(end-start,2)
                #print("TRT inference time:",infer_time,"s")
                infer_time_lst.append(infer_time)

                #post processing the image
                
                export_path=os.path.join(self.export_data_path,image.split('.')[0]+extension)
                #self.postprocess_and_save_pil(output_data,export_path)
                self.postprocess_new_jetson(export_path,output_data,self.quality)
                #print("exporting done for :",image)

                end_=time.time()
                overall_time=round(end_-start_,2)
                overall_time_lst.append(overall_time)

        elif self.input_image_mode=="L":
            
            for image in tqdm(images):

                start_=time.time()
                
                start=time.time()
                image_path=os.path.join(self.dataset_path,image)
                #image_=self.preprocess_image(image_path)
                image_=self.preprocess_gray_new_jetson_gray(image_path)
                end=time.time()
                preprocess_time=round(end-start,2)
                #print("preprocess is done for:",image,"time:",preprocess_time,"seconds")
                preprocess_time_lst.append(preprocess_time)

                #run inference
                start=time.time()
                output_data=self.infer(image_) #inference step
                end=time.time()
                
                infer_time=round(end-start,2)
                #print("TRT inference time:",infer_time,"s")
                infer_time_lst.append(infer_time)

                #post processing the image
                
                export_path=os.path.join(self.export_data_path,image.split('.')[0]+extension)
                #self.postprocess_and_save_pil(output_data,export_path)
                self.postprocess_new_jetson(export_path,output_data,self.quality)
                #print("exporting done for :",image)

                end_=time.time()
                overall_time=round(end_-start_,2)
                overall_time_lst.append(overall_time)
            
        infer_time_lst=np.array(infer_time_lst)
        overall_time_lst=np.array(overall_time_lst)
        print("Average inference time per image is:",infer_time_lst.mean(),"for dataset of size:",infer_time_lst.shape)
        print("Average overall time per image is:",overall_time_lst.mean())

        print()
        print("While exporting the image to the Disc:")
        print("Achievable FPS is:",round(1/overall_time_lst.mean(),3),"for image of resolution:",self.input_shape)


    def postprocess_new_jetson(self,output_path,output,quality):
        
        #print("shape of tensor recevied for post processing is:",output.shape) 
        
        height=self.output_shape[1]
        width=self.output_shape[2]
        
        #clip the output
        #output=np.clip(output,0,1)

        # Clip output, denormalize, and convert to uint8 in a single line
        output = (np.clip(output, 0, 1) * 255.0).astype(np.uint8).reshape(3, height, width)
        
        #denormalize and the datatype to integer type
        #output=(output*255.0).astype(np.uint8)
        
        #output=output.reshape(3,height,width)
        
        #when no cropping is done
        #image=output[0]

        #when padding cropping is to be done
        image=output[0,:-4,:]
        
        image=Image.fromarray(image,mode="L")

        #crop the extra padding by the DL model
        
        
        #exporting to the disc
        start=time.time()
        image.save(output_path,format=self.img_format,quality=quality)
        end=time.time()
        export_time=round(end-start,3)
        
    def preprocess_gray_new_jetson(self,image_path):
        
        "This function is for RGB input images"
        
        height,width=self.input_shape[1],self.input_shape[2]
        
        image=Image.open(image_path).resize((width,height)).convert("L")
        image=np.array(image)/255.0
        
        #stack the grayscale image into the first channel of a blank 3D array
        img_final=np.zeros((1,3,height,width),dtype=np.float32)
        img_final[0,0,:,:]=image
        
        return img_final.ravel()
    
    def preprocess_gray_new_jetson_gray(self,image_path):
        """This function is for the input image which is gray
        """
        height,width=self.input_shape[1],self.input_shape[2]
        
        image=Image.open(image_path).resize((width,height))
        image=np.array(image)/255.0
        
        #stack the grayscale image into the first channel of a blank 3D array
        img_final=np.zeros((1,3,height,width),dtype=np.float32)
        img_final[0,0,:,:]=image
        
        #print("shape of image after preprocessing is:",img_final.shape)

        return img_final.ravel()
        
    
    
        
        
        
    



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='DL image compression using TensorRT')
    parser.add_argument('--engine_path', required=True, type=str, help='Path of the DL model engine file')
    parser.add_argument('--dataset_path',required=True, type=str,help='Path of the dataset')
    parser.add_argument('--project_path',required=True, type=str,help='Path of the project.It may not exists yet')
    parser.add_argument('--input_shape',required=True, type=int, nargs=2,help='Input shape of the model height and width as two integers')
    obj_dl=DL_compression(parser.parse_args())
    obj_dl.main()
