#pragma once
#include "batchedtensor.hpp"
#include "helper.hpp"

using namespace tel;

// TODO make cleaner
template <std::size_t SIZE>
class Sigmoid {
private:
    BatchedTensor<float, SIZE> previous_output;
public:
	Sigmoid() : previous_output(0, nullptr) {}
	void forward(const BatchedTensor<float, SIZE>& input, BatchedTensor<float, SIZE>& output);

	void backward(BatchedTensor<float, SIZE>& error);

    ~Sigmoid(){
        delete[] previous_output.data;
    }	

};
static float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

static float sigmoid_dif(float x){
    float sig = sigmoid(x);
    return sig*(1-sig);
}

// TODO inplace
template <std::size_t SIZE>
static void sigmoid_forward_cpu(const BatchedTensor<float, SIZE> &input, BatchedTensor<float, SIZE> &output){
    size_t size = input.size();
    for (int i = 0; i < size; i++)
        output(i) = sigmoid(input(i));

}

template <std::size_t SIZE>
static void sigmoid_backward_cpu( BatchedTensor<float, SIZE> &error, const BatchedTensor<float, SIZE> &previous_output){
    size_t size = error.size();
    for (int i = 0; i < size; i++)
        error(i) = sigmoid_dif(previous_output(i))*error(i);

}

template <std::size_t SIZE>
void Sigmoid<SIZE>::forward(const BatchedTensor<float, SIZE>& input, BatchedTensor<float, SIZE>& output)
{
	sigmoid_forward_cpu(input, output);
    copy_prev(output, previous_output);
}


template <std::size_t SIZE>
void Sigmoid<SIZE>::backward(BatchedTensor<float, SIZE> &error)
{
	sigmoid_backward_cpu(error, this->previous_output);
}
