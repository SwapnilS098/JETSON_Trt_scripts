#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR) {
            std::cerr << "ERROR: " << msg << std::endl;
        } else if (severity == Severity::kWARNING) {
            std::cerr << "WARNING: " << msg << std::endl;
        } else {
            std::cout << msg << std::endl;
        }
    }
};

ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile, IBuilder* builder, IBuilderConfig* config, IRuntime* runtime, Logger& logger) {
    // Create a network definition with explicit batch
    INetworkDefinition* network = builder->createNetworkV2(1 << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // Create an ONNX parser
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse the ONNX file: " << onnxFile << std::endl;
        return nullptr;
    }

    // Create optimization profile
    IOptimizationProfile* profile = builder->createOptimizationProfile();

    // Define dynamic input shapes (example dimensions, adjust as per your model)
    const char* inputTensorName = "input";  // Use your input tensor's name here

    // Min, Opt, Max dimensions for dynamic input (for example, dynamic batch size, fixed 3x1232x1640)
    profile->setDimensions(inputTensorName, OptProfileSelector::kMIN, Dims4{1, 3, 2464, 3280}); // Min batch size = 1
    profile->setDimensions(inputTensorName, OptProfileSelector::kOPT, Dims4{1, 3, 2464, 3280}); // Optimal batch size = 4
    profile->setDimensions(inputTensorName, OptProfileSelector::kMAX, Dims4{1, 3, 2464, 3280}); // Max batch size = 8

    // Add optimization profile to the builder config
    config->addOptimizationProfile(profile);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    // Set the workspace size (can be adjusted based on available GPU memory)
    config->setMaxWorkspaceSize(1 << 28); // 256 MB (adjust as needed)

    // Build the engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to build the engine!" << std::endl;
        return nullptr;
    }

    return engine;
}

void saveEngineToFile(ICudaEngine* engine, const std::string& engineFilePath) {
    // Serialize the engine to a file
    IHostMemory* serializedModel = engine->serialize();
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();
    std::cout << "Engine saved to " << engineFilePath << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <onnx_file_path> <engine_file_path>" << std::endl;
        return -1;
    }

    const std::string onnxFile = argv[1];
    const std::string engineFilePath = argv[2];

    Logger logger;

    // Initialize the TensorRT runtime and builder
    IBuilder* builder = createInferBuilder(logger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IRuntime* runtime = createInferRuntime(logger);

    if (!builder || !config || !runtime) {
        std::cerr << "Failed to create TensorRT components!" << std::endl;
        return -1;
    }

    // Build the engine from the ONNX file
    ICudaEngine* engine = buildEngineFromOnnx(onnxFile, builder, config, runtime, logger);
    if (!engine) {
        std::cerr << "Failed to build the engine from the ONNX file!" << std::endl;
        return -1;
    }

    // Save the engine to the file
    saveEngineToFile(engine, engineFilePath);

    // Clean up
    engine->destroy();
    builder->destroy();
    config->destroy();
    runtime->destroy();

    return 0;
}

