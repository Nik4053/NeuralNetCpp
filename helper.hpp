#pragma once

#include "batchedtensor.hpp"

template <std::size_t SIZE_PREV>
void copy_prev(const tel::BatchedTensor<float, SIZE_PREV> &input, tel::BatchedTensor<float, SIZE_PREV> &previous_input){
    size_t batch_size = input.batch_size;
    // copy input to prev_input
    if (previous_input.batch_size != batch_size || previous_input.data == nullptr)
    {
        delete[] previous_input.data;
        float *data = new float[batch_size * SIZE_PREV];
        previous_input.data = data;
        previous_input.batch_size = batch_size;
    }
    // copy input to prev_input
    previous_input.set(input);
}