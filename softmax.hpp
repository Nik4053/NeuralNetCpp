#pragma once

#include "batchedtensor.hpp"
#include "helper.hpp"
#include <cmath>

using namespace tel;

template <std::size_t SIZE>
class SoftMax
{
private:
    /* data */
    BatchedTensor<float, SIZE> prev_output;
public:
    SoftMax():prev_output(0,nullptr){}
    ~SoftMax(){
        delete[] prev_output.data;
    }
    void forward(const BatchedTensor<float, SIZE>& input, BatchedTensor<float, SIZE>& output);
    void backward(BatchedTensor<float, SIZE>& error);
};

template <std::size_t SIZE>
void SoftMax<SIZE>::forward(const BatchedTensor<float, SIZE>& input, BatchedTensor<float, SIZE>& output){
    assert(input.batch_size == output.batch_size);
    size_t batch_size = input.batch_size;
    // shift input for numerical stability
    float shifted_data[batch_size*SIZE];
    BatchedTensor<float, SIZE> shifted(input.batch_size, shifted_data);
    float max = input.max();
    for (size_t i = 0; i < shifted.batch_size; i++){
        shifted[i].set(input[i]-max);
    }

    // calculate softmax
    // exp
    for (size_t i = 0; i < shifted.batch_size; i++){
        shifted[i].set(exp(shifted[i]));
    }
    // sum
    for (size_t b = 0; b < batch_size; b++)
    {
        float sum = shifted[b].sum();
        output[b].set( shifted[b] / sum ); // TODO: check for division by zero
    }
    // copy output to prev_output
    copy_prev(output, prev_output);  
}

template <std::size_t SIZE>
void SoftMax<SIZE>::backward(BatchedTensor<float, SIZE>& error){
    size_t batch_size = error.batch_size;
    float sumT_data[batch_size];
    BatchedTensor<float, 1> sumT(batch_size, sumT_data);
    sumT.set(0);
    for (size_t b = 0; b < batch_size; b++){
        for (size_t i = 0; i < SIZE; i++){
            sumT(b) += error(b, i) * prev_output(b, i);
        }
    }

    for (size_t b = 0; b < batch_size; b++){
        for (size_t i = 0; i < SIZE; i++){
            error(b, i) = prev_output(b, i) * (error(b, i) - sumT(b));
        }
    }
}
