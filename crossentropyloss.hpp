#pragma once

#include <limits>
#include <cmath>
#include "batchedtensor.hpp"
#include "helper.hpp"


using namespace tel;

template <std::size_t LABEL_NUM>
class CrossEntropyLoss
{
private:
    /* data */
    BatchedTensor<float, LABEL_NUM> prev_input;
public:
    CrossEntropyLoss():prev_input(0,nullptr){}
    ~CrossEntropyLoss(){
        delete[] prev_input.data;
    }
    float forward(const BatchedTensor<float, LABEL_NUM>& input, const BatchedTensor<float, LABEL_NUM>& target);
    void backward(BatchedTensor<float, LABEL_NUM>& target);
};

template <std::size_t LABEL_NUM>
float CrossEntropyLoss<LABEL_NUM>::forward(const BatchedTensor<float, LABEL_NUM>& input, const BatchedTensor<float, LABEL_NUM>& target){
    assert(input.batch_size == target.batch_size);
    size_t batch_size = input.batch_size;
    // copy input to prev_input
    copy_prev(input, prev_input);
    // forward
    float loss = 0;
    for (size_t b = 0; b < batch_size; b++){
        for (size_t i = 0; i < LABEL_NUM; i++){
            if( target(b,i) == 1){
                loss += -log(input(b,i) + std::numeric_limits<float>::epsilon());
            }
        }
    }
    return loss/(LABEL_NUM*batch_size);
}

template <std::size_t LABEL_NUM>
void CrossEntropyLoss<LABEL_NUM>::backward(BatchedTensor<float, LABEL_NUM>& target){
    assert(prev_input.batch_size == target.batch_size);
    // prev_input.print();
    size_t batch_size = target.batch_size;
    for (size_t b = 0; b < batch_size; b++){
        for (size_t i = 0; i < LABEL_NUM; i++){
            if( target(b,i) == 1){
                target(b,i) = -1.0f / (prev_input(b,i) + std::numeric_limits<float>::epsilon());
            } else {
                target(b,i) = 0;
            }
        }
    }
}