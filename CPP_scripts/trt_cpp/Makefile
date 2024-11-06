
all:
	
	g++ -o build_engine build_engine.cpp \
    -I/usr/local/cuda/include \
    -I/usr/include/aarch64-linux-gnu \
    -I/usr/include/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/aarch64-linux-gnu \
    -lnvinfer -lnvonnxparser -lcuda -lstdc++


#	g++ -o build_engine_python build_engine_python.cpp \
    -I/usr/include \
    -I/usr/local/cuda/include \
    -I/usr/include/onnx \
    -L/usr/lib/aarch64-linux-gnu \
    -lnvinfer -lnvonnxparser -lcudart -lonnxruntime

	g++ -o trt_infer_cpp trt_infer_cpp.cpp -lnvinfer -lcudart -lcuda -lstdc++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64

