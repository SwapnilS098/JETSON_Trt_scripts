#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include "logger.h"

// Precision options
enum class Precision {
    FP32,
    FP16,
    INT8,
};

struct Options {
    Precision precision = Precision::FP32;
    int optBatchSize = 1;
    int maxBatchSize = 16;
    int deviceIndex = 0;
    int64_t maxWorkspaceSize = 1 << 30; // 1GB
    std::string calibrationDataDir;
};

class Logger : public nvinfer1::ILogger {
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

ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile, const Options& options, Logger& logger) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ERROR: Failed to parse ONNX file." << std::endl;
        return nullptr;
    }

    // Optimization profile for dynamic batch sizes
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{options.optBatchSize, 3, 224, 224});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{options.maxBatchSize, 3, 224, 224});
    config->addOptimizationProfile(profile);

    // Set precision mode
    if (options.precision == Precision::FP16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (options.precision == Precision::INT8 && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // Add INT8 calibrator here if required
    }

    config->setMaxWorkspaceSize(options.maxWorkspaceSize);

    // Build engine
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "ERROR: Engine build failed." << std::endl;
        return nullptr;
    }

    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    return engine;
}

void saveEngineToFile(nvinfer1::ICudaEngine* engine, const std::string& filePath) {
    nvinfer1::IHostMemory* serializedEngine = engine->serialize();
    std::ofstream file(filePath, std::ios::binary);
    file.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    file.close();
    serializedEngine->destroy();
    std::cout << "Engine saved at: " << filePath << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <onnx_file_path> <engine_file_path> <precision_mode (FP32/FP16/INT8)>" << std::endl;
        return -1;
    }

    const std::string onnxFile = argv[1];
    const std::string engineFilePath = argv[2];
    const std::string precisionStr = argv[3];

    Precision precision;
    if (precisionStr == "FP32") precision = Precision::FP32;
    else if (precisionStr == "FP16") precision = Precision::FP16;
    else if (precisionStr == "INT8") precision = Precision::INT8;
    else {
        std::cerr << "ERROR: Unknown precision mode " << precisionStr << std::endl;
        return -1;
    }

    Logger logger;
    Options options = {precision};

    nvinfer1::ICudaEngine* engine = buildEngineFromOnnx(onnxFile, options, logger);
    if (!engine) {
        std::cerr << "ERROR: Failed to build engine from ONNX model." << std::endl;
        return -1;
    }

    saveEngineToFile(engine, engineFilePath);
    engine->destroy();
    return 0;
}

