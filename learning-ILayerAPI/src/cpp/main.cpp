#include <iostream>
#include <memory>

#include "utils.hpp"
#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    // Model model("models/weights/sample_linear.weights");
    // Model model("models/weights/sample_conv.weights");
    // Model model("models/weights/sample_permute.weights");
    // Model model("models/weights/sample_reshape.weights");
    // Model model("models/weights/sample_batchNorm.weights");
    // Model model("models/weights/sample_cbr.weights");
    // Model model("models/weights/sample_pooling.weights");
    // Model model("models/weights/sample_upsample.weights");
    // Model model("models/weights/sample_deconv.weights");
    // Model model("models/weights/sample_concat.weights");
    // Model model("models/weights/sample_elementwise.weights");
    // Model model("models/weights/sample_reduce.weights");
    Model model("models/weights/sample_slice.weights");

    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
    if(!model.infer()){
        LOGE("fail in infering model");
        return 0;
    }
    return 0;
}
