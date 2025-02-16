#pragma once

#include "batchedtensor.hpp"
#include "helper.hpp"

using namespace tel;

template <std::size_t SIZE>
class ReLu
{
private:
    /* data */
    BatchedTensor<float, SIZE> prev_input;
public:
    ReLu():prev_input(0,nullptr){}
    ~ReLu(){
        delete[] prev_input.data;
    }
    void forward(const BatchedTensor<float, SIZE>& input, BatchedTensor<float, SIZE>& output);
    void backward(BatchedTensor<float, SIZE>& error);
};

template <std::size_t SIZE>
void ReLu<SIZE>::forward(const BatchedTensor<float, SIZE>& input, BatchedTensor<float, SIZE>& output){
    assert(input.batch_size == output.batch_size);
    size_t batch_size = input.batch_size;
    // copy input to prev_input
    copy_prev(input, prev_input);
    // forward
    for (size_t b = 0; b < batch_size; b++){
        for (size_t i = 0; i < SIZE; i++){
            output(b, i) = std::max(0.0f, input(b, i));
        }
    }
}

template <std::size_t SIZE>
void ReLu<SIZE>::backward(BatchedTensor<float, SIZE>& error){
    for (size_t b = 0; b < prev_input.batch_size; b++){
        for (size_t i = 0; i < SIZE; i++){
            if(prev_input(b,i) <= 0){
                error(b,i) = 0;
            }
        }
    }
}
