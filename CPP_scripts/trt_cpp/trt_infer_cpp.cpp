#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

nvinfer1::ICudaEngine* loadEngine(const std::string& engineFilePath, nvinfer1::ILogger& logger) {
    std::ifstream engineFile(engineFilePath, std::ios::binary);
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

    // Deserialize engine (still using deprecated API for now, consider updating to V2 in the future)
    return runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
}

void doInference(nvinfer1::ICudaEngine* engine, const void* inputData, void* outputData, int batchSize) {
    assert(engine != nullptr);

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Get input and output sizes
    size_t inputSize = engine->getBindingDimensions(0).d[0] * engine->getBindingDimensions(0).d[1] *
                        engine->getBindingDimensions(0).d[2] * engine->getBindingDimensions(0).d[3];
    size_t outputSize = engine->getBindingDimensions(1).d[0] * engine->getBindingDimensions(1).d[1] *
                         engine->getBindingDimensions(1).d[2] * engine->getBindingDimensions(1).d[3];

    // Use cudaMallocManaged with void** cast to avoid invalid conversion errors
    float* d_input;
    float* d_output;
    cudaMallocManaged((void**)&d_input, inputSize * sizeof(float));   // Cast to void**
    cudaMallocManaged((void**)&d_output, outputSize * sizeof(float)); // Cast to void**

    cudaMemcpy(d_input, inputData, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    void* buffers[2] = {d_input, d_output};
    context->executeV2(buffers);

    cudaMemcpy(outputData, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
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

    // Assuming inputData and outputData have been initialized correctly
    std::vector<float> inputData(1 * 3 * 2464 * 3280);
    std::vector<float> outputData(1 * 1000);

    doInference(engine, inputData.data(), outputData.data(), 1);

    return 0;
}

