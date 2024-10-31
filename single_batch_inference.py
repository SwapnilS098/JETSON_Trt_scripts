
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
import torch
from PIL import Image
import time
import cv2
import os
import onnx


from build_engine import Build_engine
print("Build_engine class is imported")


from trt_inference import TensorRTInference
print("TensorRTInference class is imported")



def trt_main(onnx_model_name,onnx_path_base,engine_path_base,input_shape,output_shape,dataset_path,export_data_path,engine_name):

    #final onnx model path
    onnx_path=os.path.join(onnx_path_base,onnx_model_name+".onnx")

    #final engine model path
    if engine_name=="same":
        engine_path=os.path.join(engine_path_base,onnx_model_name+".engine")
    else:
        engine_name=onnx_model_name+"_"+engine_name
        engine_path=os.path.join(engine_path_base,engine_name+".engine")
    print("engine path is:",engine_path)

    #loading or building the engine
    start=time.time()
    Engine=Build_engine(onnx_path,engine_path,input_shape)

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

    #Now running the inference on the whole dataset

    #instantiating the TensorRTInference class
    trt_inference=TensorRTInference(engine_path,dataset_path,export_data_path,input_shape,output_shape,gray) #output_path is the image output path

    q_cr=trt_inference.inference_over_dataset()
    print("quality cr list:",q_cr)



    if __name__=="__main__":

    #onnx_model_name="bmshj_halfUHD_ssim_6" #write the name without the extension
    #onnx_model_name="bmshj_halfUHD_ssim_4"
    #onnx_model_name="bmshj_4_UHD_org"
    onnx_model_name="bmshj_4_UHD"
    #onnx_model_name="bmshj4_UHD_gray_org_version"
    #onnx_model_name="bmshj_4_UHD_org"
    print("getting the information about the onnx model")


    engine_name="JETSON"   #if some optimization parameter is added then can write here #it will append this name to the onnx_model name while exporting
    #input_shape=[3,1232,1640]
    #output_shape=[3,1232,1648]  #take care of the 8 pixels added by the model in the output shape
    #input_shape=[3,720,1280]
    #output_shape=[3,720,1280]
    #input_shape=[3,2464,3280]
    #output_shape=[3,2464,3280]
    input_shape=[3,2464,3280]
    output_shape=[3,2464,3280]
    gray=False

    onnx_path_base=r"/home/swapnil09/DL_comp_final_09_24/onnx_scripts/onnx_export/onnx_models"
    onnx_path_complete=os.path.join(onnx_path_base,onnx_model_name+".onnx")

    onnx_model=onnx.load(onnx_path_complete)


    engine_path_base=r"/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts/engine_models"
    #dataset_path=r"/home/swapnil09/DL_comp_final_09_24/Dataset_50/dataset_50_gray"
    dataset_path=r"/home/swapnil09/DL_comp_final_09_24/Dataset_50/Dataset_50"
    export_data_path=r"/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts_gray/trt_infer_output"
