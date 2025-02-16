#pragma once
#include "batchedtensor.hpp"

using namespace tel;


template <std::size_t DIM, std::size_t... DIMS>
class Flatten {
private:
    typedef Tensor<float, DIM, DIMS...> input_type;
    typedef Tensor<float, input_type::SIZE> output_type;
    typedef BatchedTensor<float, DIM, DIMS...> input_type_batched;
    typedef BatchedTensor<float, input_type::SIZE> output_type_batched;
public:

	Flatten(){}


	output_type_batched forward(const input_type_batched& input);

	input_type_batched backward(const output_type_batched& error);

};


template <std::size_t DIM, std::size_t... DIMS>
typename Flatten<DIM, DIMS... >::output_type_batched Flatten<DIM, DIMS... >::forward(const input_type_batched &input)
{
	output_type_batched output(input.batch_size,input.data);
    return output;
}


template <std::size_t DIM, std::size_t... DIMS>
typename Flatten<DIM, DIMS... >::input_type_batched Flatten<DIM, DIMS... >::backward(const output_type_batched &error)
{
	input_type_batched input(error.batch_size,error.data);
    return input;
}
