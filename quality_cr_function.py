

"""
    Here we are trying to write a function that can 
    fit the compression ratio and the quality parameter of the 
    codecs such as JPEG by using various compression operations over the images dataset.

"""

import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.preprocessing import PolynomialFeatures
import io

MODEL_PATH="quality_compression_model.pkl"
def calculate_compression_ratio(image,quality):
    """
    compress the image to a specific quality and return the compression ratio
    """

    #compress image to a buffer
    buffer=io.BytesIO()
    image.save(buffer,format="JPEG",quality=quality)

    #calculate original and compressed size
    original_size=len(image.tobytes())
    compressed_size=buffer.tell()

    compression_ratio=original_size/compressed_size

    return compression_ratio


def build_quality_compression_model(dataset_path):

    """
    builds a regression model to predict the JPEG quality from the compression ratio"""
    cr_quality_pairs=[]

    #load image and process
    print("building the model")
    for image_file in sorted(os.listdir(dataset_path)):
        if image_file.lower().endswith((".jpg",".webp",".png",".webp")):
            image_path=os.path.join(dataset_path,image_file)
            image=Image.open(image_path).convert("RGB")

        for quality in range(1,100,1):
            cr=calculate_compression_ratio(image,quality)
            cr_quality_pairs.append([cr,quality])
        print("done for image:",image_file)

    compression_ratios=np.array([pair[0] for pair in cr_quality_pairs]).reshape(-1,1)
    qualities=np.array([pair[1] for pair in cr_quality_pairs])

    #fit a polynomial regression model
    model=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())

    model.fit(compression_ratios,qualities)
    print("model is fit")
    
    #saving the model
    joblib.dump(model,MODEL_PATH)
    print("model is exported")

    return model

def load_or_build_model(dataset_path):

    """
    loads the model
    """
    if os.path.exists(MODEL_PATH):
        print("Loading the model from the path:")
        model=joblib.load(MODEL_PATH)
    else:
        print("Model not found hence building the model")
        model=build_quality_compression_model(dataset_path)

    return model 


def predict_quality(model,compression_ratio):

    predicted_quality=model.predict(np.array([[compression_ratio]]))
    return max(0,min(100,int(predicted_quality[0]))) #clamp between 0 and 100


dataset_path=r"/home/swapnil09/DL_comp_final_09_24/Dataset_50/Dataset_50"
#model=build_quality_compression_model(dataset_path)
model=load_or_build_model(dataset_path)

compression_ratio=45
import time

start=time.time()
predicted_quality=predict_quality(model,compression_ratio)
end=time.time()
print("time in prediction:",round(end-start,3),"seconds")
print("predicted quality is:",predicted_quality,"for the compression ratio of :",compression_ratio)


