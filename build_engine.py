
"""
    Dont run this script directly
    It will be called using the trt_main script.

    It builds the engine file for the onnx file of the model
    and it loads the file if it exists on the disc"""

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

        #get the information about the model loaded from the engine file
        print("Input shapes:")
        for i in range(engine.num_io_tensors):
            tensor_name=engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name)==trt.TensorIOMode.INPUT:
                shape=engine.get_tensor_shape(tensor_name)
                print(f"Name: {tensor_name}, shape: {shape}")
        print("Output shapes:")
        for i in range(engine.num_io_tensors):
            tensor_name=engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name)==trt.TensorIOMode.OUTPUT:
                shape=engine.get_tensor_shape(tensor_name)
                print(f"Name : {tensor_name}, Shape: {shape}")



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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<33) 
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
        strip_weights=False
        if strip_weights:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)
        #to remove strip plan from config
        #config.flags&=~(1<<int(trt.BuilderFlag.STRIP_PLAN))

        #config.clear_flag(trt.BuilderFlag.TF32)

        #config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

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
        #Using the 3 channel gray version 
        profile.set_shape(input_tensor.name,(1,3,self.height,self.width),
                          (1,3,self.height,self.width),
                          (1,3,self.height,self.width))
        
        #building for the grayscale version where the channel is changed from 3 to 1
        #profile.set_shape(input_tensor.name,(1,1,self.height,self.width),
        #                    (1,1,self.height,self.width),
        #                    (1,1,self.height,self.width))


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
