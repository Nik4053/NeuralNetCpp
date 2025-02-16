#pragma once
#include "tensor.hpp"

using namespace tel;

template <std::size_t SIZE_W, std::size_t SIZE_B >
class OptimizerSGD{
    private:
    public:
        float learning_rate;
        OptimizerSGD(float learning_rate){
            this->learning_rate = learning_rate;
        }
        void update(Tensor<float,SIZE_W> weights, Tensor<float,SIZE_B> bias, Tensor<float,SIZE_W> gradient_weights, Tensor<float,SIZE_B> bias_gradients){
            for(size_t i = 0; i < SIZE_W; i++){
                weights[i] -= learning_rate * gradient_weights[i];
            }
            for(size_t i = 0; i < SIZE_B; i++){
                bias[i] -= learning_rate * bias_gradients[i];
            }
        }
};