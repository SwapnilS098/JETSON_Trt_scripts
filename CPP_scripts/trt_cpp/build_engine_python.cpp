#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <memory>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
};

class BuildEngine {
public:
    BuildEngine(const std::string& onnxFile, const std::string& engineFile, int height, int width)
        : onnxPath(onnxFile), enginePath(engineFile), height(height), width(width) {}

    nvinfer1::ICudaEngine* loadEngine(nvinfer1::IRuntime* runtime) {
        std::ifstream file(enginePath, std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        return runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
    }

    nvinfer1::ICudaEngine* buildEngine() {
        Logger logger;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

        // Read ONNX model into a buffer
        std::ifstream modelFile(onnxPath, std::ios::binary);
        modelFile.seekg(0, std::ios::end);
        size_t modelSize = modelFile.tellg();
        modelFile.seekg(0, std::ios::beg);

        std::vector<char> modelData(modelSize);
        modelFile.read(modelData.data(), modelSize);

        if (!parser->parse(modelData.data(), modelSize)) {
            std::cerr << "Failed to parse ONNX model." << std::endl;
            return nullptr;
        }

        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);  // 1 GB for workspace

        // Set dimensions for optimization profile
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        auto inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, height, width));
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, height, width));
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, height, width));

        config->addOptimizationProfile(profile);
        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        return engine;
    }

private:
    std::string onnxPath;
    std::string enginePath;
    int height;
    int width;
};

int main() {
    std::string onnxFile = "model.onnx";
    std::string engineFile = "model.trt";
    int height = 224, width = 224;

    BuildEngine builder(onnxFile, engineFile, height, width);
    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = builder.loadEngine(runtime);

    if (engine == nullptr) {
        std::cerr << "Failed to load the engine!" << std::endl;
        return 1;
    }

    std::cout << "Engine loaded successfully!" << std::endl;
    return 0;
}

