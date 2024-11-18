//
// Created by sebas on 17.11.2024.
//
#include <matrix.hpp>
#include <numbers>
#include<print>
#include <sigmoid.hpp>
#include<vector.hpp>
using namespace math;

void sigmoid_test() {
    Matrix in = {
        {-0.5, -0.40909091, -std::numbers::inv_pi, -0.22727273},
        {-0.13636364, -0.04545455, 0.04545455, 0.13636364},
        {0.22727273, std::numbers::inv_pi, 0.40909091, 0.5}
    };

    Matrix out_truth = {
        {0.37754067, 0.39913012, 0.42111892, 0.44342513},
        {0.46596182, 0.48863832, 0.51136168, 0.53403818},
        {0.55657487, 0.57888108, 0.60086988, 0.62245933}
    };

    Matrix dout_truth = {
        {-0.11750186, -0.09811034, -0.07756566, -0.05609075},
        {-0.03393292, -0.01135777, 0.01135777, 0.03393292},
        {0.05609075, 0.07756566, 0.09811034, 0.11750186},
    };

    layer::Sigmoid sigmoid = layer::Sigmoid(in.n(), in.m());
    Matrix* out = sigmoid.forward(std::move(in));
    out->print();
    Matrix dout = sigmoid.backward(std::move(in));
    dout.print();
}

int main() {
    sigmoid_test();
    // float dot_product = vector / vector2;
    // std::println("Dot: {}", dot_product);
}
