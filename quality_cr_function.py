import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib  # For saving and loading the model
import io

MODEL_PATH = "quality_compression_model.pkl"

def calculate_compression_ratio(image,image_size, quality):
    """
    Compress an image to a specific JPEG quality and return the compression ratio.
    the images are read from the disc
    """
    #image=Image.open(image_path).convert("RGB")
    # Compress image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    
    # Calculate original and compressed size
    original_size = image_size#os.path.getsize(image_path)
    compressed_size = buffer.tell()  # in byteis
    #print("org: size:",original_size,"comp_size:",compressed_size)
    
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size
    buffer.close()
    return compression_ratio, original_size

def build_quality_compression_model(dataset_path):
    """
    Builds a regression model to predict JPEG quality from input image size and compression ratio.
    """
    cr_quality_pairs = []
    print("starting the model fitting")
    # Load and process each image
    for image_file in sorted(os.listdir(dataset_path)):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = os.path.join(dataset_path, image_file)
            #print("image path:",image_path)
            image = Image.open(image_path).convert("RGB")
            image_size=os.path.getsize(image_path)

            # Test quality settings and collect compression ratios and input sizes
            for quality in range(1, 100, 1):
                compression_ratio, input_size = calculate_compression_ratio(image,image_size, quality)
                cr_quality_pairs.append([input_size, compression_ratio, quality])
                print("quality:",quality)
                #print("cr:",compression_ratio)
        print("done for the image :",image_file)

    # Prepare data for fitting
    input_sizes = np.array([pair[0] for pair in cr_quality_pairs]).reshape(-1, 1)
    compression_ratios = np.array([pair[1] for pair in cr_quality_pairs]).reshape(-1, 1)
    qualities = np.array([pair[2] for pair in cr_quality_pairs])

    # Combine input size and compression ratio into a single feature array
    X = np.hstack((input_sizes, compression_ratios))

    # Fit a polynomial regression model for multiple features
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    model.fit(X, qualities)

    # Save the model to disk
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model

def load_or_build_model(dataset_path):
    """
    Loads the model from disk if available, otherwise builds and saves it.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    else:
        print("Model not found, building a new model...")
        model = build_quality_compression_model(dataset_path)
    
    return model

def predict_quality(model, input_size, compression_ratio):
    """
    Predicts JPEG quality for a given input size and compression ratio using the model.
    """
    # Combine input size and compression ratio for prediction
    X_pred = np.array([[input_size, compression_ratio]])
    predicted_quality = model.predict(X_pred)
    
    # Clamp the result between 0 and 100 for JPEG quality scale
    return max(0, min(100, int(predicted_quality[0])))

# Usage example
dataset_path = r"/home/swapnil09/DL_comp_final_09_24/Dataset_50/Dataset_50"  # Set your dataset path
model = load_or_build_model(dataset_path)

# Example input image size and compression ratio for prediction
input_size = 1000  # Example input size in bytes
compression_ratio = 2.5  # Example compression ratio
predicted_quality = predict_quality(model, input_size, compression_ratio)
print(f"Predicted JPEG quality for input size {input_size} and compression ratio {compression_ratio}: {predicted_quality}")

