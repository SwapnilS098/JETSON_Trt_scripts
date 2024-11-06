import time
import io
import asyncio
import random
from PIL import Image
import single_batch_inference as single
import glob

def load_image(image_path):
    """Loads an image from disk and converts it to a buffer."""
    image = Image.open(image_path)
    buffer = io.BytesIO()
    image.save(buffer, format=image.format)
    buffer.seek(0)
    return buffer

async def async_infer_full_pipeline(image_buffer, engine_path, input_shape, output_shape, image_index):
    """Run full pipeline asynchronously for each image: preprocess, inference, postprocess, export."""
    inference_object = single.Inference(engine_path, input_shape, output_shape, image_buffer)
    
    # Step 1: Preprocess
    preprocess_start = time.time()
    preprocessed_image, image_size = inference_object.preprocess_gray_2(image_buffer)
    preprocess_end = time.time()
    print(f"Image {image_index} - Preprocessing time: {preprocess_end - preprocess_start:.3f} seconds")

    # Step 2: Inference
    inference_start = time.time()
    output_data = inference_object.infer(preprocessed_image)
    inference_end = time.time()
    print(f"Image {image_index} - Inference time: {inference_end - inference_start:.3f} seconds")

    # Step 3: Postprocess
    postprocess_start = time.time()
    quality = 20  # AVIF encoding quality parameter
    img_size, output_image_buffer = inference_object.postprocess_gray_final(output_data, quality)
    postprocess_end = time.time()
    print(f"Image {image_index} - Postprocessing time: {postprocess_end - postprocess_start:.3f} seconds")
    
    # Step 4: Export
    export_start = time.time()
    export_image(output_image_buffer, f"exported_image_{image_index + 1}.avif")
    export_end = time.time()
    print(f"Image {image_index} - Export time: {export_end - export_start:.3f} seconds")

    return output_image_buffer

async def process_images_async(image_paths, engine_path, input_shape, output_shape):
    tasks = []
    for i, image_path in enumerate(image_paths[:10]):  # Process the first 10 images
        # Simulate random delay before each image is processed
        delay = random.expovariate(1/0.01)  # Mean delay of 0.1 seconds
        await asyncio.sleep(delay)  # Introduce delay

        # Load image as buffer and start async inference for the full pipeline
        image_buffer = load_image(image_path)
        print(f"Starting full pipeline for image {i+1} with delay: {delay:.3f} seconds")
        start=time.time()
        task = async_infer_full_pipeline(image_buffer, engine_path, input_shape, output_shape, i)
        end=time.time()
        print("time per image:",round(end-start,4),"seconds")
        print()
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    return results

def export_image(output_image_buffer, export_path):
    """Exports image buffer to the specified path."""
    with open(export_path, "wb") as f:
        f.write(output_image_buffer.getvalue())
    print(f"Image exported to {export_path}")

if __name__ == "__main__":
    # Set up paths and parameters
    dataset_path = r"/home/swapnil09/DL_comp_final_09_24/Dataset_50/Dataset_50"
    engine_path = r"/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts/engine_models/bmshj_4_UHD_JETSON.engine"
    input_shape = [3, 2464, 3280]
    output_shape = [3, 2464, 3280]

    # Gather first 30 images from dataset
    image_paths = glob.glob(f"{dataset_path}/*.png")[:10]  # Adjust extension if necessary

    # Run asynchronous image processing
    start_time = time.time()
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_images_async(image_paths, engine_path, input_shape, output_shape))

    print(f"Processed {len(results)} images in {time.time() - start_time:.2f} seconds.")

