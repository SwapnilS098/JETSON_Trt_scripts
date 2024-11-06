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
    // Create a network definition
    INetworkDefinition* network = builder->createNetworkV2(0);

    // Create an ONNX parser
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse the ONNX file: " << onnxFile << std::endl;
        return nullptr;
    }

    // Configure the builder for FP32 precision (default, no need to call setFp16Mode or setInt8Mode)
    builder->setMaxBatchSize(1);  // The batch size is inferred from the model or the input

    // We no longer need to set workspace size manually as TensorRT will handle this automatically
    // builder->setMaxWorkspaceSize(1 << 28);  // Set workspace to 256MB (adjust this based on available memory)

    // Build the engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to build the engine!" << std::endl;
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

