#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                      << " at line " << __LINE__ << " in file " << __FILE__;   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }



nvinfer1::ICudaEngine* loadEngine(const std::string& engineFilePath, nvinfer1::ILogger& logger) {
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    std::cout<<"Load Engine function starting"<<std::endl;
    if (!engineFile.is_open()) {
        std::cerr << "Error opening engine file: " << engineFilePath << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Error creating TensorRT runtime" << std::endl;
        return nullptr;
    }
	std::cout<<"Load engine function completed runnning:"<<std::endl;
    return runtime->deserializeCudaEngine(engineData.data(), engineSize);
}

void doInference(nvinfer1::ICudaEngine* engine, const void* inputData, void* outputData, int batchSize) {
    assert(engine != nullptr);
    std::cout << "doInference function is starting..." << std::endl;

    //std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    std::unique_ptr>context(engine->createExecutionContext());
    assert(context != nullptr);

    // Get input and output binding dimensions
    nvinfer1::Dims inputDims = context->getBindingDimensions(0);
    nvinfer1::Dims outputDims = context->getBindingDimensions(1);

    size_t inputSize = batchSize;
    for (int i = 0; i < inputDims.nbDims; i++) inputSize *= inputDims.d[i];
    size_t outputSize = batchSize;
    for (int i = 0; i < outputDims.nbDims; i++) outputSize *= outputDims.d[i];

    std::cout << "Input size: " << inputSize << ", Output size: " << outputSize << std::endl;

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), outputSize * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, inputData, inputSize * sizeof(float), cudaMemcpyHostToDevice));

    void* buffers[2] = {d_input, d_output};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run inference
    bool status = context->enqueueV2(buffers, stream, nullptr);
    if (!status) {
        std::cerr << "Error: Enqueue failed." << std::endl;
        return;
    }

    // Copy output data from device to host
    CUDA_CHECK(cudaMemcpy(outputData, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "doInference function completed." << std::endl;
}

int main() {
    std::string enginePath = "/home/swapnil09/DL_comp_final_09_24/tensorrt_scripts_gray/tensorRT_scripts/CPP_scripts/trt_cpp/bmshj_4_UHD_JETSON_CPP.engine";
    class Logger : public nvinfer1::ILogger {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
            if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
                std::cout << msg << std::endl;
            }
        }
    } logger;

    nvinfer1::ICudaEngine* engine = loadEngine(enginePath, logger);
    if (engine == nullptr) {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }

    // Initialize input and output data (ensure dimensions are correct)
    std::vector<float> inputData(1 * 3 * 2464 * 3280);
    std::vector<float> outputData(1 * 1000);

    doInference(engine, inputData.data(), outputData.data(), 1);

    engine->destroy();
    return 0;
}

