#pragma once

#include "batchedtensor.hpp"
#include "optimizersgd.hpp"
#include "helper.hpp"


using namespace tel;

template <std::size_t SIZE_PREV, std::size_t SIZE>
class Dense{
    private:
    BatchedTensor<float, SIZE_PREV> previous_input;
    float weights_data[SIZE_PREV*SIZE];
    float bias_data[SIZE];
    // gradients
    float gradients_weights_data[SIZE_PREV*SIZE];
    float gradients_bias_data[SIZE];

    public:
    Tensor<float,SIZE_PREV,SIZE> weights; // (size_previous_layer, size_this_layer)
    Tensor<float,SIZE> bias; // (  size_this_layer )
    OptimizerSGD<SIZE_PREV*SIZE,SIZE>* optimizer;
    Dense( OptimizerSGD<SIZE_PREV*SIZE,SIZE>* optimizer = nullptr):optimizer(optimizer),previous_input(0,nullptr),weights(weights_data),bias(bias_data){
        
    }
    ~Dense(){
        delete[] previous_input.data;
    }
    void forward(const BatchedTensor<float, SIZE_PREV>& input, BatchedTensor<float, SIZE>& output){
        size_t batch_size = input.batch_size;
        // copy input to previous_input
        copy_prev(input, previous_input);

        // forward
        for (size_t b = 0; b < batch_size; b++){
            for (size_t i = 0; i < SIZE; i++){
                float weighted_sum = bias(i);
                for (size_t j = 0; j < SIZE_PREV; j++){
                    weighted_sum += input(b, j) * weights(j, i);
                }
                output(b, i) = weighted_sum;
            }
        }
    }
    void backward(const BatchedTensor<float, SIZE>& error, BatchedTensor<float, SIZE_PREV>& error_output);
    void initialize(float min, float max);
};

template <std::size_t SIZE_PREV, std::size_t SIZE>
void Dense<SIZE_PREV, SIZE>::initialize(float min, float max){
    float range = max - min;
    std::srand(1234);
    for (int w = 0; w < weights.SIZE; w++){
        weights.flatten()(w) = (double)std::rand()/RAND_MAX * range + min;
    }
    for (int b = 0; b < bias.SIZE; b++){
        bias(b) = (double)std::rand()/RAND_MAX * range + min;
    }
}

template <std::size_t SIZE_PREV, std::size_t SIZE>
void Dense<SIZE_PREV, SIZE>::backward(const BatchedTensor<float, SIZE>& error, BatchedTensor<float, SIZE_PREV>& error_output){
    assert(previous_input.batch_size == error.batch_size);
    assert(error_output.batch_size == error.batch_size);
    size_t batch_size = error.batch_size;

    // calculate error_output
    for (size_t b = 0; b < batch_size; b++){
        for (size_t j = 0; j < SIZE_PREV; j++){
            double tmp = 0;
            for (size_t i = 0; i < SIZE; i++){
                tmp += error(b, i) * weights(j, i);
            }
            error_output(b, j) = tmp;
        }
    }
    

    if (optimizer == nullptr) return;

    // calculate gradients
    // weights
    Tensor<float,SIZE_PREV,SIZE> gradients_weights(gradients_weights_data);
    for (size_t j = 0; j < SIZE_PREV; j++){
        for (size_t i = 0; i < SIZE; i++){
            float tmp = 0;
            for (size_t b = 0; b < batch_size; b++){
                tmp += error(b, i) * previous_input(b, j);
            }
            gradients_weights(j, i) = tmp;
        }
    }
    // bias
    Tensor<float,SIZE> gradients_bias(gradients_bias_data);
    for (size_t i = 0; i < SIZE; i++){
        float tmp = 0;
        for (size_t b = 0; b < batch_size; b++){
            tmp += error(b, i);
        }
        gradients_bias(i) = tmp;
    }

    // update weights and bias
    optimizer->update(weights, bias, gradients_weights, gradients_bias);

}