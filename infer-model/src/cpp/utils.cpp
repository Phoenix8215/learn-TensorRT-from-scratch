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
    if(!cache.good()) {
        LOGE("can't open %s", path.c_str());
        return false;
    }

    /* 将engine的内容写入trtModelStream中*/
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();

    // vector<uint8_t> tmp;
    trtModelStream.seekg(0, ios::beg);
    // tmp.resize(size);

    /* 将trtModelStream中的stream通过read函数写入modelMem中*/
    trtModelStream.read(reinterpret_cast<char *>(data.data()), size);
    return true;
}

/* 常规的使用iostream读取文件的方法，可复用*/
vector<unsigned char> loadFile(const string &file){
    vector<uint8_t> data;

    /* 通过ifstream读取文件，并保存为unsigned char的vector*/
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()) return {};

    /* 设置数据流位置到末尾，并获取文件大小*/
    in.seekg(0, ios::end);
    size_t length = in.tellg();
    if (length <= 0) return {};

    /* 通过ifstream读取文件，并保存为unsigned char的vector*/
    in.seekg(0, ios::beg);
    data.resize(length);
    in.read(reinterpret_cast<char *>(data.data()), length);

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

