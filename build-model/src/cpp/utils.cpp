#include "utils.hpp"
#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include "NvInfer.h"

using namespace std;

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))) {
        return false;
    } else {
        return true;
    }
}

/**
 * @brief 获取engine的大小size，并将engine的信息载入到data中，
 * 
 * @param path engine的路径
 * @param data 存储engine数据的vector
 * @param size engine的大小
 * @return true 文件读取成功
 * @return false 文件读取失败
 */
bool fileRead(const string &path, vector<unsigned char> &data, size_t &size){
    stringstream trtModelStream;
    ifstream cache(path);
    if(!cache.is_open()) {
        cerr << "Unable to open file😅: " << path << endl;
        return false;
    }

    /* 将engine的内容写入trtModelStream中*/
    trtModelStream.seekg(0, trtModelStream.beg);
    trtModelStream << cache.rdbuf();
    cache.close();

    /* 计算model的大小*/
    trtModelStream.seekg(0, ios::end);
    size = trtModelStream.tellg();
    data.resize(size);

    // vector<uint8_t> tmp;
    trtModelStream.seekg(0, ios::beg);
    // tmp.resize(size);

    // read方法将从trtModelStream读取的数据写入以data[0]为起始地址的内存位置
    trtModelStream.read(reinterpret_cast<char *>(data.data()), size);
    return true;
}


vector<unsigned char> loadFile(const string &file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<unsigned char> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read(reinterpret_cast<char*>(data.data()), length);
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

// models/onnx/sample.onnx
string getEnginePath(string onnxPath){
    int name_l = onnxPath.rfind("/");
    int name_r = onnxPath.rfind(".");

    int dir_r  = onnxPath.find("/");

    string enginePath;
    enginePath = onnxPath.substr(0, dir_r);
    enginePath += "/engine";
    enginePath += onnxPath.substr(name_l, name_r - name_l);
    enginePath += ".engine";
    return enginePath;
}


// string getEnginePath(string onnxPath) {
//     // 使用 std::filesystem 来处理文件路径
//     experimental::filesystem::path filePath(onnxPath);

//     // 获取父目录路径
//     experimental::filesystem::path parentPath = filePath.parent_path();

//     // 获取文件名（不包括扩展名）
//     std::string filename = filePath.stem().string();

//     // 构造引擎文件路径
//     experimental::filesystem::path enginePath = parentPath / "engine" / (filename + ".engine");

//     return enginePath.string();
// }
