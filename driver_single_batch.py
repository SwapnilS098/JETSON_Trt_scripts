
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
    #print("infer_ function is working")
    start=time.time()
    image_buffer=load_image(image_path)
    end=time.time()
    print("image loading time from disc:",round(end-start,3),"seconds")
    print()
    #print("type of objectL:",type(image_buffer))

    start=time.time()
    inference_object=single.Inference(engine_path,input_shape,output_shape,image_buffer)
    end=time.time()
    print("engine preparation time is:",round(end-start,3),"seconds")
    print()

    start=time.time()
    output_image_buffer=inference_object.main()
    end=time.time()
    print("image inference time is:",round(end-start,3),"seconds")

    return output_image_buffer

def export_image(output_image_buffer,export_path):
    with open(export_path,"wb") as f:
        f.write(output_image_buffer.getvalue())
    print(f"Image exported to {export_path}")


if __name__=="__main__":
    image_path="image.png"
    export_path="exported_image.jpg"
    input_shape=[3,2464,3280]
    output_shape=[3,2464,3280]
    engine_path=r"/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts/engine_models/bmshj_4_UHD_JETSON.engine"

    output_image_buffer=infer_(image_path,engine_path,input_shape,output_shape)
    start=time.time()
    export_image(output_image_buffer,export_path)
    end=time.time()
    print("image export time to disc in the binary writing form is:",round(end-start,3),"seconds")
    print()
