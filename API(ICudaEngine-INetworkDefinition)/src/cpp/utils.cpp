#include <experimental/filesystem>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "NvInfer.h"


using namespace std;

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))){
        return false;
    }else{
        return true;
    }
}

bool fileRead(const string &path, vector<unsigned char> &data, size_t &size){
    stringstream trtModelStream;
    ifstream cache(path);

    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();

    trtModelStream.seekg(0, ios::beg);

    // trtModelStream.read((char*)&data[0], size);
    trtModelStream.read(reinterpret_cast<char *>(data.data()), size);
    return true;
}

vector<unsigned char> loadFile(const string &file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read(reinterpret_cast<char *>(data.data()), length);
    }
    in.close();
    return data;
}

std::string printDims(const nvinfer1::Dims dims) {
    std::ostringstream oss;
    oss << "[ ";
    for (int i = 0; i < dims.nbDims; i++) {
        oss << dims.d[i];
        if (i != dims.nbDims - 1) {
            oss << ", ";
        }
    }
    oss << " ]";
    return oss.str();
}

std::string printTensor(float* tensor, int size) {
    std::ostringstream oss;
    oss << "[ ";
    for (int i = 0; i < size; i++) {
        oss << std::fixed << std::setprecision(4) << tensor[i];
        if (i != size - 1) {
            oss << ", ";
        }
    }
    oss << " ]";
    return oss.str();
}


string printTensorShape(nvinfer1::ITensor* tensor){
    string str;
    str += "[";
    auto dims = tensor->getDimensions();
    for (int j = 0; j < dims.nbDims; j++) {
        str += to_string(dims.d[j]);
        if (j != dims.nbDims - 1) {
            str += " x ";
        }
    }
    str += "]";
    return str;
}

string getEnginePath(string onnxPath){
    int name_l = onnxPath.rfind("/");
    int name_r = onnxPath.rfind(".");

    int dir_l  = 0;
    int dir_r  = onnxPath.find("/");

    string enginePath;
    enginePath = onnxPath.substr(dir_l, dir_r);
    enginePath += "/engine";
    enginePath += onnxPath.substr(name_l, name_r - name_l);
    enginePath += ".engine";
    return enginePath;
}

string getPrecision(nvinfer1::DataType type) {
    switch(type) {
        case nvinfer1::DataType::kFLOAT:  return "FP32";
        case nvinfer1::DataType::kHALF:   return "FP16";
        case nvinfer1::DataType::kINT32:  return "INT32";
        case nvinfer1::DataType::kINT8:   return "INT8";
        default:                          return "unknown type❓";
    }
}


string getWhere(nvinfer1::TensorLocation locate) {
    switch(locate) {
        case nvinfer1::TensorLocation::kDEVICE: return "on device";
        case nvinfer1::TensorLocation::kHOST:   return "on host";
        default:                                return "error location❓";
    }

}

std::string getLayerType(nvinfer1::LayerType type) {
    switch(type) {
        case nvinfer1::LayerType::kCONVOLUTION:        return "Convolution";
        case nvinfer1::LayerType::kFULLY_CONNECTED:    return "Fully Connected";
        case nvinfer1::LayerType::kACTIVATION:         return "Activation";
        case nvinfer1::LayerType::kPOOLING:            return "Pooling";
        case nvinfer1::LayerType::kLRN:                return "LRN";
        case nvinfer1::LayerType::kSCALE:              return "Scale";
        case nvinfer1::LayerType::kSOFTMAX:            return "SoftMax";
        case nvinfer1::LayerType::kDECONVOLUTION:      return "Deconvolution";
        case nvinfer1::LayerType::kCONCATENATION:      return "Concatenation";
        case nvinfer1::LayerType::kELEMENTWISE:        return "Elementwise";
        case nvinfer1::LayerType::kPLUGIN:             return "Plugin";
        case nvinfer1::LayerType::kUNARY:              return "Unary";
        case nvinfer1::LayerType::kPADDING:            return "Padding";
        case nvinfer1::LayerType::kSHUFFLE:            return "Shuffle";
        case nvinfer1::LayerType::kREDUCE:             return "Reduce";
        case nvinfer1::LayerType::kTOPK:               return "TopK";
        case nvinfer1::LayerType::kGATHER:             return "Gather";
        case nvinfer1::LayerType::kMATRIX_MULTIPLY:    return "Matrix Multiply";
        case nvinfer1::LayerType::kRAGGED_SOFTMAX:     return "Ragged Softmax";
        case nvinfer1::LayerType::kCONSTANT:           return "Constant";
        case nvinfer1::LayerType::kRNN_V2:             return "RNNv2";
        case nvinfer1::LayerType::kIDENTITY:           return "Identity";
        case nvinfer1::LayerType::kPLUGIN_V2:          return "PluginV2";
        case nvinfer1::LayerType::kSLICE:              return "Slice";
        case nvinfer1::LayerType::kSHAPE:              return "Shape";
        case nvinfer1::LayerType::kPARAMETRIC_RELU:    return "Parametric ReLU";
        case nvinfer1::LayerType::kRESIZE:             return "Resize";
        case nvinfer1::LayerType::kTRIP_LIMIT:         return "Trip Limit";
        case nvinfer1::LayerType::kRECURRENCE:         return "Recurrence";
        case nvinfer1::LayerType::kITERATOR:           return "Iterator";
        case nvinfer1::LayerType::kLOOP_OUTPUT:        return "Loop Output";
        case nvinfer1::LayerType::kSELECT:             return "Select";
        case nvinfer1::LayerType::kFILL:               return "Fill";
        case nvinfer1::LayerType::kQUANTIZE:           return "Quantize";
        case nvinfer1::LayerType::kDEQUANTIZE:         return "Dequantize";
        case nvinfer1::LayerType::kCONDITION:          return "Condition";
        case nvinfer1::LayerType::kCONDITIONAL_INPUT:  return "Conditional Input";
        case nvinfer1::LayerType::kCONDITIONAL_OUTPUT: return "Conditional Output";
        case nvinfer1::LayerType::kSCATTER:            return "Scatter";
        case nvinfer1::LayerType::kEINSUM:             return "Einsum";
        case nvinfer1::LayerType::kASSERTION:          return "Assertion";
        case nvinfer1::LayerType::kONE_HOT:            return "One Hot";
        case nvinfer1::LayerType::kNON_ZERO:           return "Non Zero";
        case nvinfer1::LayerType::kGRID_SAMPLE:        return "Grid Sample";
        case nvinfer1::LayerType::kNMS:                return "NMS";
        default:                                       return "Unknown Type❓";
    }
}

std::string getTensorMode(nvinfer1::TensorIOMode type) {
    switch(type) {
        case nvinfer1::TensorIOMode::kINPUT:    return "is input✔️";
        case nvinfer1::TensorIOMode::kOUTPUT:   return "is output✔️";
        default:                                return "is not an input or output❌";
    }
}
