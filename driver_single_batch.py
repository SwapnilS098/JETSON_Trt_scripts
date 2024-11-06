
import time
import io
import single_batch_inference as single
from PIL import Image

def load_image(image_path):
    image=Image.open(image_path)
    buffer=io.BytesIO()
    image.save(buffer,format=image.format)
    buffer.seek(0)
    return buffer

def infer_(image_path,engine_path,input_shape,output_shape):
    print("infer_ function is working")
    image_buffer=load_image(image_path)
    print("type of objectL:",type(image_buffer))
    inference_object=single.Inference(engine_path,input_shape,output_shape,image_buffer)
    output_image_buffer=inference_object.main()
    return output_image_buffer

def export_image(output_image_buffer,export_path):
    with open(export_path,"wb") as f:
        f.write(output_image_buffer.getvalue())
    print(f"Image exported to {export_path}")


if __name__=="__main__":
    image_path="image.png"
    export_path="exported_image.avif"
    input_shape=[3,2464,3280]
    output_shape=[3,2464,3280]
    engine_path=r"/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts/engine_models/bmshj_4_UHD_JETSON.engine"

    output_image_buffer=infer_(image_path,engine_path,input_shape,output_shape)
    export_image(output_image_buffer,export_path)

