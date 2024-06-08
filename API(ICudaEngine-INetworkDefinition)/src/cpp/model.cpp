#include <cstring>
#include <memory>
#include <iostream>
#include <string>
#include <type_traits>

#include "model.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "cuda_runtime.h"

using namespace std;

class Logger : public nvinfer1::ILogger{
public:
    virtual void log (Severity severity, const char* msg) noexcept override{
        string str;
        switch (severity){
            case Severity::kINTERNAL_ERROR: str = RED    "[fatal]" CLEAR; break;
            case Severity::kERROR:          str = RED    "[error]" CLEAR; break;
            case Severity::kWARNING:        str = BLUE   "[warn]"  CLEAR; break;
            case Severity::kINFO:           str = YELLOW "[info]"  CLEAR; break;
            case Severity::kVERBOSE:        str = PURPLE "[verb]"  CLEAR; break;
        }
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;

Model::Model(string onnxPath){
    if (!fileExists(onnxPath)) {
        LOGE("%s not found. Program terminated", onnxPath.c_str());
        exit(1);
    }
    mOnnxPath = onnxPath;
    mEnginePath = getEnginePath(mOnnxPath);
}

bool Model::build(){
    if (fileExists(mEnginePath)){
        LOG("%s has been generated!", mEnginePath.c_str());
        return true;
    } else {
        LOG("%s not found. Building engine...", mEnginePath.c_str());
    }
    Logger logger;
    auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    config->setMaxWorkspaceSize(1<<28);

    if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
        LOGE("ERROR: failed to %s", mOnnxPath.c_str());
        return false;
    }

    auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    std::ofstream f(mEnginePath, std::ios::binary);
    f.write(plan->data(), plan->size());
    f.close();

    mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    mInputDims         = network->getInput(0)->getDimensions();
    mOutputDims        = network->getOutput(0)->getDimensions();
    LOG("Input dim is %s", printDims(mInputDims).c_str());
    LOG("If it is network input❓:%d", network->getInput(0)->isNetworkInput());
    LOG("Input Tensors type is %s", getPrecision(network->getInput(0)->getType()).c_str());
    LOG("Input Tensors is locate %s", getWhere(network->getInput(0)->getLocation()).c_str());

    LOG("Output dim is %s", printDims(mOutputDims).c_str());
    LOG("If it is network output❓:%d", network->getOutput(0)->isNetworkOutput());
    LOG("Output Tensors type is %s", getPrecision(network->getInput(0)->getType()).c_str());
    LOG("Output Tensors is locate %s", getWhere(network->getOutput(0)->getLocation()).c_str());

    // 把优化前和优化后的各个层的信息打印出来
    LOG("Before TensorRT optimization💩");
    print_network(*network, false);
    LOG("");
    LOG("After TensorRT optimization🚀");
    print_network(*network, true);
    return true;
};

bool Model::infer(){
    /*
        我们在infer需要做的事情
        1. 读取model => 创建runtime, engine, context
        2. 把数据进行host->device传输
        3. 使用context推理
        4. 把数据进行device->host传输
    */

    /* 1. 读取model => 创建runtime, engine, context */
    if (!fileExists(mEnginePath)) {
        LOGE("ERROR: %s not found", mEnginePath.c_str());
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(mEnginePath);
    
    Logger logger;
    auto runtime     = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine      = make_unique<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    auto context     = make_unique<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    auto input_dims   = context->getBindingDimensions(0);
    auto output_dims  = context->getBindingDimensions(1);

    LOG("input dim shape is:  %s", printDims(input_dims).c_str());
    LOG("output dim shape is: %s", printDims(output_dims).c_str());

    /* 2. host->device的数据传递 */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* host memory上的数据*/
    float input_host[]{0.0193, 0.2616, 0.7713, 0.3785, 0.9980, 0.9008, 0.4766, 0.1663, 0.8045, 0.6552};
    float output_host[5];

    /* device memory上的数据*/
    float* input_device = nullptr;
    float* weight_device = nullptr;
    float* output_device = nullptr;

    int input_size = 10;
    int output_size = 5;

    /* 分配空间, 并传送数据从host到device*/
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMalloc(&output_device, sizeof(output_host));
    cudaMemcpyAsync(input_device, input_host, sizeof(input_host), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 3. 模型推理, 最后做同步处理 */
    float* bindings[] = {input_device, output_device};
    bool success = context->enqueueV2((void**)bindings, stream, nullptr);

    /* 4. device->host的数据传递 */
    cudaMemcpyAsync(output_host, output_device, sizeof(output_host), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    LOG("input data is:  %s", printTensor(input_host, input_size).c_str());
    LOG("output data is: %s", printTensor(output_host, output_size).c_str());
    LOG("finished inference");
    return true;
}

void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized) {
    // ITensor, ILayer, INetwork
    // ICudaEngine, IExecutionContext, IBuilder

    int32_t inputCount = network.getNbInputs();
    LOG("Input count is %d", inputCount);

    int32_t outputCount = network.getNbOutputs();
    LOG("Input count is %d", outputCount);

    LOG("Lays count is %d", network.getNbLayers());

    LOG("network name is %s", network.getName());

    LOG("if implicit batch ❓%d", network.hasImplicitBatchDimension());

    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        nvinfer1::ITensor* input = network.getInput(i);
        LOG("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        nvinfer1::ITensor* output = network.getOutput(i);
        LOG("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }

    // 注意这里指的是网络的原始层数和TensorRT优化后的层数
    // private成员:std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    int layerCount = optimized ? mEngine->getNbLayers() : network.getNbLayers();
    LOG("network has %d layers", layerCount);

    if (!optimized) { // 没有优化走这个分支
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            nvinfer1::ILayer* layer = network.getLayer(i);
            nvinfer1::ITensor* input = layer->getInput(0);

            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOG("🌟 layer_info: %-20s | Type: %-15s | Input Shape: %-15s | Output Shape: %-15s | Precision: [%s]",
                layer->getName(),
                getLayerType(layer->getType()).c_str(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        size_t device_size = mEngine->getDeviceMemorySize();
        LOG("device memory size is %d.🚀", device_size);
        LOG("network name is %s", mEngine->getName());
        LOG("if has implicit batch dimension❓", mEngine->hasImplicitBatchDimension());

        // 输出 or 输出 or 其他
        LOG("get tensor1 mode %s", getTensorMode(mEngine->getTensorIOMode("input0")).c_str());
        LOG("get tensor2 mode %s", getTensorMode(mEngine->getTensorIOMode("/linear/MatMul")).c_str());

        // 返回指定Tensor的组件个数和每个组件的大小，单位是字节
        LOG("input has %d elements, the size of element is %d Bytes", mEngine->getTensorComponentsPerElement("input0"),\
                                                                        mEngine->getTensorBytesPerComponent("input0"));

        LOG("Tensor format is %s", mEngine->getTensorFormatDesc("input0"));

        LOG("Vectorized tensor's index is %d", mEngine->getTensorVectorizedDim("input0"));
        auto inspector = make_unique<nvinfer1::IEngineInspector>(mEngine->createEngineInspector());

        string info = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kONELINE);
        LOG("layer_info: %s", info.c_str());

        for (int i = 0; i < layerCount; i++) {
            string info = inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kONELINE);

            if(!info.empty()) {
                info.pop_back(); // 去掉字符串末尾的换行符
            }
            LOG("layer_info: %s", info.c_str());
        }
    }
}
