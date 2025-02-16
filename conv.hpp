#pragma once

#include <iostream>
#include "tensor.hpp"
#include "batchedtensor.hpp"
#include "optimizersgd.hpp"

using namespace tel;
// TODO: Conv should not be Height and Width dependent
template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
class Conv2D {
private:
    typedef Tensor<float, OUT_C, IN_C, filter_size, filter_size > WeightTensor;
    typedef Tensor<float, OUT_C> BiasTensor;
	BatchedTensor<float,OUT_C,H,W> previous_output;
	BatchedTensor<float,IN_C,H,W> last_input;

    float *weights_data = new float[WeightTensor::SIZE];
    float *bias_data = new float [BiasTensor::SIZE];
	void convolution_forward_cpu(const BatchedTensor<float, IN_C, H, W> &input, BatchedTensor<float, OUT_C, H, W> &output, const WeightTensor &weights, const BiasTensor &bias);
    void convolution_backward_cpu(const BatchedTensor<float, OUT_C, H, W> &error, BatchedTensor<float, IN_C, H, W> &next_error, const WeightTensor &weights);
	void convolution_gradient_weights_reduction_cpu(const BatchedTensor<float,IN_C,H,W> &input, const BatchedTensor<float,OUT_C,H,W> &error,  WeightTensor &gradient_weights);
	void convolution_gradient_bias_reduction_cpu(const BatchedTensor<float, OUT_C, H, W> &error, BiasTensor &gradient_bias);
public:
	OptimizerSGD<WeightTensor::SIZE,BiasTensor::SIZE>* optimizer;
	WeightTensor weights;
	BiasTensor bias;
	Conv2D(OptimizerSGD<WeightTensor::SIZE,BiasTensor::SIZE>* optimizer): optimizer(optimizer), weights(weights_data), bias(bias_data), previous_output(0,nullptr), last_input(0,nullptr)  {	}
	
	void forward(const BatchedTensor<float, IN_C, H, W>& input, BatchedTensor<float, OUT_C, H, W>& output);
	void backward(const BatchedTensor<float, OUT_C, H, W>& error, BatchedTensor<float, IN_C, H, W>& error_output);
	void initialize(float min, float max);
    ~Conv2D(){
        delete[] weights_data;
        delete[] bias_data;
		delete[] previous_output.data;
		delete[] last_input.data;
    }
};

template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void Conv2D<IN_C,OUT_C,filter_size,H,W>::convolution_forward_cpu(const BatchedTensor<float, IN_C, H, W> &input, BatchedTensor<float, OUT_C, H, W> &output, const WeightTensor &weights, const BiasTensor &bias) {

	int batch_size = input.getDim(0);

	for (int c_out = 0; c_out < OUT_C; c_out++){ 
		float channel_bias = bias(c_out);
		for (int b = 0; b < batch_size; b++) {
			for (int y = 0; y < H; y++) {
				for (int x = 0; x < W; x++) {
					float val = channel_bias;
					for (int c_in = 0; c_in < IN_C; c_in++) {
						// apply stencil
						for (int j = (-filter_size+1)/2; j <= (filter_size)/2; j++) {
							for (int i = (-filter_size+1)/2; i <= (filter_size)/2; i++) {
								float input_val = (x+i>=0&&x+i<W&&y+j>=0&&y+j<H)?input(b,c_in,y+j,x+i):0;
								val += input_val * weights(c_out, c_in, j + (filter_size-1) / 2, i+(filter_size-1)/2);
							}
						}
					}
					output(b,c_out,y,x) = val;
				}
			}
		}
	}
}


template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void Conv2D<IN_C,OUT_C,filter_size,H,W>::convolution_backward_cpu(const BatchedTensor<float, OUT_C, H, W> &error, BatchedTensor<float, IN_C, H, W> &next_error, const WeightTensor &weights) {

	int batch_size = error.getDim(0);


	for (int c_in = 0; c_in < IN_C; c_in++) {
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				for (int b = 0; b < batch_size; b++) {
					float val = 0.;
					for (int c_out = 0; c_out < OUT_C; c_out++) {
						for (int j = (-filter_size+1)/2; j <= (filter_size)/2; j++) {
							for (int i = (-filter_size+1)/2; i <= (filter_size)/2; i++) {
								float error_val =0;
								if(x + i >= 0 && x + i < W && y + j >= 0 && y + j < H) {
									error_val=error(b, c_out, y + j, x + i);
								} 
								val += error_val * weights(c_out, c_in, filter_size/2 - j , filter_size/2 - i );
							}
						}
					}
					next_error(b, c_in, y, x) = val;
				}
			}
		}
	}
}

template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void Conv2D<IN_C,OUT_C,filter_size,H,W>::convolution_gradient_weights_reduction_cpu(const BatchedTensor<float,IN_C,H,W> &input, const BatchedTensor<float,OUT_C,H,W> &error,  WeightTensor &gradient_weights) {

	const short batch_size = input.getDim(0);


	// channel
	for (int c_out = 0; c_out < OUT_C; c_out++) {		
		for (int c_in = 0; c_in < IN_C; c_in++) {	
			// stencil
			for (short j = (-filter_size+1)/2; j <= filter_size/2; j++) {
				for (short i = (-filter_size+1)/2; i <= filter_size/2; i++) {
					float val = 0.;
					// pixel
					for (int y = 0; y < H; y++) {
						for (int x = 0; x < W; x++) {
							for (short b = 0; b < batch_size; b++) {
								float input_val =0;
								if (x + i >= 0 && x + i < W && y + j >= 0 && y + j < H) {
									input_val=input(b, c_in, y + j, x + i); // for each image go through the same filter for one single channel
								} 
								val += input_val * error(b, c_out, y, x);
							}
						}
					}
					gradient_weights(c_out, c_in, j+(filter_size-1)/2, i+(filter_size-1)/2) = val;								
				}
			}
		}
	}

}

template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void  Conv2D<IN_C,OUT_C,filter_size,H,W>::convolution_gradient_bias_reduction_cpu(const BatchedTensor<float, OUT_C, H, W> &error, BiasTensor &gradient_bias)
{
	const int batch_size = error.getDim(0),
			  output_channels = error.getDim(1),
			  height = error.getDim(2),
			  width = error.getDim(3);



	for (int c_out = 0; c_out < output_channels; c_out++) {	
		float val = 0.;
		for (int b = 0; b < batch_size; b++)
			for (int y = 0; y < height; y++) 
				for (int x = 0; x < width; x++) 
					val += error(b, c_out, y, x);
		gradient_bias(c_out) += val;
	}
}




template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void Conv2D<IN_C,OUT_C,filter_size,H,W>::initialize(float min, float max)
{
	float range = max - min;
	std::srand(1234);
	std::cout << "Initializing weights and bias" << std::endl;
	std::cout << "RandomValue: " << std::rand() << std::endl;
	for (int out_c = 0; out_c < weights.getDim(0); out_c++)
		for (int c = 0; c < weights.getDim(1); c++)
			for (int y = 0; y < weights.getDim(3); y++)
				for (int x = 0; x < weights.getDim(2); x++)
					weights(out_c, c, y, x) = (double)std::rand()/RAND_MAX * range + min;

	for (int f = 0; f < bias.getDim(0); f++)
		bias(f) = (double)std::rand()/RAND_MAX * range + min;
}
template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void Conv2D<IN_C,OUT_C,filter_size,H,W>::forward(const BatchedTensor<float, IN_C, H, W> &input, BatchedTensor<float, OUT_C, H, W> &output)
{
	// std::cout << "ConvCPU::forward" << std::endl;

	size_t batch_size = input.batch_size;

    // check if input size is the same as last input size
    if (this->last_input.getDim(0) != batch_size || this->last_input.data == nullptr) {
		if (this->last_input.data != nullptr) delete[] this->last_input.data;
        float* data_in = new float[batch_size*IN_C*H*W];
		this->last_input.batch_size = batch_size;
		this->last_input.data = data_in;
    }
	this->last_input.set(input);
	
    // check if output size is the same as last output size
    if (this->previous_output.getDim(0) != batch_size || this->previous_output.data == nullptr) {
		if (this->previous_output.data != nullptr) delete[] this->previous_output.data;
        float* data_out = new float[batch_size*OUT_C*H*W];
		this->previous_output.batch_size = batch_size;
		this->previous_output.data = data_out;
    }
	convolution_forward_cpu(input, output, weights, bias);
	this->previous_output.set(output);

}




template<size_t IN_C, size_t OUT_C, int filter_size, size_t H, size_t W>
void Conv2D<IN_C,OUT_C,filter_size,H,W>::backward(const BatchedTensor<float, OUT_C, H, W> &error, BatchedTensor<float, IN_C, H, W> &error_output)
{
	
	size_t batch_size = error.getDim(0);

    float gradiant_weights_data[WeightTensor::SIZE];
    float gradiant_bias_data[BiasTensor::SIZE];

    WeightTensor gradient_weights(gradiant_weights_data);
    BiasTensor gradient_bias(gradiant_bias_data);
	gradient_weights.setZero();
	gradient_bias.setZero();


	
	convolution_gradient_weights_reduction_cpu(this->last_input, error, gradient_weights);

	convolution_gradient_bias_reduction_cpu(error, gradient_bias);

	optimizer->update(weights, bias, gradient_weights, gradient_bias);

	convolution_backward_cpu(error, error_output, weights);
}