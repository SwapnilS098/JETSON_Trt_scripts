"""
    Dont run this script
    Run it using the trt_main script

    The postprocess_save and postprocess_save_pil are two versions of the
    postprocessing code.
    postprocess_save uses the opencv which is giving better results

    While the postprocess_save_pil is creating the color distortions in the
    output image particularly in the green color
    """



import tensorrt as trt
print(trt.__version__)
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import pillow_avif
from PIL import Image
import time
import cv2
import os
import matplotlib.pyplot as plt

class TensorRTInference:
    def __init__(self, engine_path,dataset_path,export_data_path,input_shape,output_shape,gray):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.dataset_path=dataset_path
        self.export_data_path=export_data_path
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.gray=gray #boolean
        

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

    def image_handling(self):
        lst=os.listdir(self.dataset_path)
        images=[]
        for image in lst:
            if image.lower().endswith("jpg") or image.lower().endswith("jpeg") or image.lower().endswith("png") or image.lower().endswith("webp"):
                images.append(image)
        #print("dataset has:",len(images),"images")
        images.sort() #sorting the images

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
        #print("Images are:",images)
        #print()
        print()
        #print("NOT SAVING THE PROCESSED IMAGE FOR NOW")
        for image in images:

            start_=time.time()
            
            start=time.time()
            image_path=os.path.join(self.dataset_path,image)
            #image_=self.preprocess_image_cv(image_path) # preprocess for the gray image
            #image_=self.preprocess_image(image_path)
            image_=self.preprocess_gray_1(image_path)
            #image_=self.preprocess_image_cv(image_path)

            print("Input image is:",np.array(image_).shape)
            #image_=self.preprocess_image_cv(image_path)
            #image_=self.preprocess_gray(image_path)
            end=time.time()
            #print("image_ type:",type(image_),"shape:",image_.shape)
            preprocess_time=round(end-start,2)
            print("preprocess is done for:",image,"time:",preprocess_time,"seconds")
            preprocess_time_lst.append(preprocess_time)

            #run inference
            start=time.time()
            output_data=self.infer(image_) #inference step
            end=time.time()
            #print("Output size:",len(list(output_data)),"for Image:",image)
            
            #visualize the output_data
            
            
            infer_time=round(end-start,2)
            #print("TRT inference time:",infer_time,"s")
            infer_time_lst.append(infer_time)

            #post processing the image
            export_path=os.path.join(self.export_data_path,image.split('.')[0]+".jpg")
            #self.postprocess_and_save_new(output_data,export_path)
            self.postprocess_gray_final(output_data,export_path)
            #self.postprocess_new(output_data)
            #self.postprocess_and_save(output_data,export_path)
            print("Done for :",image)
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

         
    

    def postprocess_new(self,image):
        print("image type:",type(image),"size of array:",image.shape)

        #the array has dimension of [1,2464,3280]
        #convert it to [2464,3280]

        #image=image.squeeze(0)
        image=image.reshape((1232,1640))

        #make the datatype as int
        image=image*255.0
        image=image.astype(np.uint8)
        print(image)
        image=Image.fromarray(image)
        print("now image type is:",image)
        print("show the image:")
        plt.imshow(image,cmap='gray')
        plt.show()
        #image.show()

    def preprocess_image(self,image_path):
        #import cProfile
        height=self.input_shape[1]
        width=self.input_shape[2]
        #print("H:",height,"W:",width)
        #with cProfile.Profile() as pr:
        #image=cv2.imread(image_path)
        image=Image.open(image_path).convert("L")
        image=np.array(image)
        #print("type:",type(image))
        #image=cv2.resize(image,(width,height))
        #image=np.resize(image,(3,width,height))
        #print("shape:",image.shape)
        image=image.astype(np.float32)/255.0
        #image=image.transpose(2,0,1)
        image=np.expand_dims(image,axis=(0,1))

        #pr.print_stats()
        return image.ravel()
 
       

    def preprocess_gray(self,image_path):
        """
        This is the method for the 3 channel color gray format
        where the first channel has the gray data and the other 
        channels are zero
        """
        height=self.input_shape[1]
        width=self.input_shape[2]
        #print("H:",height,"W:",width)
        
        #open the image as grayscale
        image=Image.open(image_path)#.convert("L")

        #resize the image
        #image=image.resize((width,height))

        #convert the image to numpy and normalize
        image=np.array(image)/255.0

        image=image[:,:,0]

        #create an empty numpy array
        img_blank=np.zeros((height,width,3))
        
        #assign the first channel as the gray image
        img_blank[:,:,0]=image

        img_final=np.expand_dims(img_blank,axis=0).astype(np.float32)

        #make the transpose to the shape of 1,3,2464,3280
        image_final=np.transpose(img_final,(0,3,1,2))
        
        image_final=image_final.ravel()

        return image_final

    
    def preprocess_gray_1(self,image_path):

        height,width=self.input_shape[1],self.input_shape[2]

        image = np.array(Image.open(image_path).resize((width, height)).convert("L")) / 255.0

        # Stack the grayscale image into the first channel of a blank 3D array
        img_final = np.zeros((1, 3, height, width), dtype=np.float32)
        img_final[0, 0, :, :] = image

        # Flatten and return
        return img_final.ravel()

    def postprocess_gray_final(self,output,output_path):
        """
            This is the function for the postprocessing and the saving 
            of the image in the gray form which is of the 3 channel.
            The results will be in the 3 channel with the first channel contains the 
            data and the rest of them will have be zero"""
        
        height=self.output_shape[1]
        width=self.output_shape[2]
       
        #clip the output
        output=np.clip(output,0,1)

        #denormalization and the datatype to integer type
        output=(output*255.0).astype(np.uint8)

        #then clip the output
        #output=np.clip(output,0,1)

        output=output.reshape(3,height,width)

        #print("type:",output,"shape:",output.shape)

        #now the data must be present in the first channel only 
        #lets copy the data of the first channel in the other two channels and 
        #then export the image to the disc

        #output[:,:,1]=output[:,:,0]
        #output[:,:,2]=output[:,:,0]
        #print("final image shape :",output.shape)

        image=output[0]
        
        #output=np.transpose(output,(1,2,0))

        image=Image.fromarray(image,mode='L')
        image.save(output_path,format="JPEG",quality=30)
        print("image is exported")
        









    def postprocess_and_save(self,output,output_path):
        """
        this is giving the correct output
        """
        height=self.output_shape[1]
        width=self.output_shape[2]
        #height=2464
        #width=3280
        print("H:",height,"W:",width)
        output=np.clip(output,0,1)
        #output=output.reshape(3,height,width)
        #output=output.transpose(1,2,0) #convert to the HWC format
        output=(output*255.0).astype(np.uint8)
        print("size of the output is:",output.shape)
        output=output.reshape(height,width)
        plt.imshow(output)
        plt.show()
        img=Image.fromarray(output)
        img=img.convert("L")
        img.save(output_path,format="JPEG",quality=10)
        #self.gray=False
        if self.gray==False:
            #cv2.imwrite(output_path,output)
            img=Image.fromarray(output)
            img=img.convert("L")
            img.save(output_path,format="JPEG",quality=10)
        else:
            output=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
            cv2.imwrite(output_path,output,params=[cv2.IMWRITE_JPEG_QUALITY, 20])

    def postprocess_and_save_new(self, output, output_path):
        """
        Processes the output and saves it to the specified path.
        """
        height = self.output_shape[1]
        width = self.output_shape[2]
    
        print("H:", height, "W:", width)
    
        # Ensure output is within the range [0, 1] and convert to 8-bit format
        output = np.clip(output, 0, 1)
        output = (output * 255.0).astype(np.uint8)
    
        print("Size of the output is:", output.shape)

        # Reshape the output to the specified height and width
        if len(output.shape) == 3 and output.shape[0] == 3:  # RGB format
            output = output.transpose(1, 2, 0)  # Convert to HWC format
        output = output.reshape(height, width)

        # Display the image
        plt.imshow(output, cmap='gray' if self.gray else 'viridis')
        plt.show()

        # Save the image
        img = Image.fromarray(output)
    
        if self.gray:
            img = img.convert("L")
            img.save(output_path, format="JPEG", quality=20)  # Save as grayscale JPEG
        else:
            img.save(output_path, format="JPEG", quality=10)  # Save as RGB JPEG

        print(f"Image saved to {output_path}")


    def postprocess_and_save_pil(self,output,output_path):
        """
        This is generating the distorted color of green shade"""
        
        height,width=self.output_shape[1:]
        #print("H",height,"W:",width)
        
        output=np.clip(output,0,1)
        print("the output from the model is as follows:")
        print("type:",type(output))
        print("shape:",output.shape)
        output=output.reshape(3,height,width)
        #output=output.transpose(1,2,0) #convert to the HWC format
        print("Now the shape is:",output.shape)
        output=(output*255.0).astype(np.uint8)

        img=Image.fromarray(output)
        #img.show()

        if not self.gray:
            img.save(output_path)
        else:
            img=img.convert("L")
            img.save(output_path,quality=20)


    def postprocess_and_save_pil_new(self, output,export_path):
        """
        Postprocesses the output and displays the image inline in a Jupyter notebook.
        """
        height, width = self.output_shape[1:]

    # Reshape and convert to HWC format
        output = output.reshape(3, height, width)
        output = output.transpose(1, 2, 0)  # Convert to HWC format (Height, Width, Channels)

    # Convert from BGR to RGB format (if needed)
        output = output[..., ::-1]  # Reversing the last axis to switch from BGR to RGB

    # Scale to 255 and convert to uint8
        output = (output * 255.0).astype(np.uint8)

    # Display image inline using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(output)
        plt.axis('off')  # Hide axes for cleaner display
        plt.show()

    # Convert to PIL Image (optional if further processing is needed)
        img = Image.fromarray(output)
        return img 
        
