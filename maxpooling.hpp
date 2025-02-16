#pragma once
#include "tensor.hpp"
#include "batchedtensor.hpp"
// TODO: Padding

using namespace tel;

template<size_t C, size_t H, size_t W, size_t stride>
class MaxPooling {
public:
    constexpr static size_t OUT_H = H / stride;
    constexpr static size_t OUT_W = W / stride;
private:
    BatchedTensor<unsigned int, C,OUT_H,OUT_W> maximasX;
    BatchedTensor<unsigned int, C,OUT_H,OUT_W> maximasY;
    /*
    Has same size as input. Stores the location of the maximas as 1 entries. Rest is 0
    */
    //Tensor<unsigned int, 4>* maximas = nullptr; // size of input
    void maxpooling_forward_cpu(const BatchedTensor<float, C, H, W> &input, BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasX, BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasY, BatchedTensor<float, C, OUT_H, OUT_W> &output);
    void maxpooling_backward_cpu(const BatchedTensor<float, C, OUT_H, OUT_W> &error, const BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasX, const BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasY, BatchedTensor<float, C,H,W> &next_error);
public:
	const int pool_size = 2;
    // const int stride = 2;

    MaxPooling():maximasX(0,nullptr),maximasY(0,nullptr){}


	void forward(const BatchedTensor<float, C, H, W>& input, BatchedTensor<float, C, OUT_H, OUT_W>& output);
    // TODO: Implement skip connections
	void backward(const BatchedTensor<float, C, OUT_H, OUT_W>& error, BatchedTensor<float, C, H, W>& error_output);

    ~MaxPooling(){
        delete[] maximasX.data;
        delete[] maximasY.data;
    }

};



template<size_t C, size_t H, size_t W, size_t stride>
void MaxPooling<C,H,W,stride>::maxpooling_forward_cpu(const BatchedTensor<float, C, H, W> &input, BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasX, BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasY, BatchedTensor<float, C, OUT_H, OUT_W> &output) {
	size_t batch_size = input.getDim(0);



    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < C; c++)
        {
            // move to next pixel with distance stride
            for (size_t y_in = 0; y_in < H; y_in += stride)
            {
                for (size_t x_in = 0; x_in < W; x_in += stride)
                {
                    size_t max_x_in = x_in;
                    size_t max_y_in = y_in;
                    float max_val = input(b, c, y_in, x_in);
                    // do pooling
                    for (size_t py = 0; py < pool_size; py++)
                    {
                        for (size_t px = 0; px < pool_size; px++)
                        {
                            float c_val = 0;
                            if (y_in + py >= H || x_in + px >= W) {
                                // out of bounds
                            }
                            else {
                                c_val = input(b, c, y_in + py, x_in + px);
                            }
                            if (c_val>max_val)
                            {
                                max_val = c_val;
                                max_x_in = x_in + px;
                                max_y_in = y_in + py;
                            }
                        }
                    }
                    output(b, c, y_in / stride, x_in / stride) = max_val;
                    //maximas(b, c, max_y_in, max_x_in) = 1;
                    maximasX(b, c, y_in / stride, x_in / stride) = max_x_in;
                    maximasY(b, c, y_in / stride, x_in / stride) = max_y_in;
                }
            }
        }
    }

}

template<size_t C, size_t H, size_t W, size_t stride>
void MaxPooling<C,H,W,stride>::maxpooling_backward_cpu(const BatchedTensor<float, C, OUT_H, OUT_W> &error, const BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasX, const BatchedTensor<unsigned int, C,OUT_H,OUT_W>& maximasY, BatchedTensor<float, C,H,W> &next_error) {
	int batch_size = error.getDim(0);
    
    for (size_t b = 0; b < batch_size; b++)
    {
        for (size_t c = 0; c < C; c++)
        {
            for (size_t y_out = 0; y_out < OUT_H; y_out++)
            {
                for (size_t x_out = 0; x_out < OUT_W; x_out++)
                {
                    int x_in = maximasX(b, c, y_out, x_out);
                    int y_in = maximasY(b, c, y_out, x_out);
                    next_error(b, c, y_in, x_in) = error(b, c, y_out, x_out);
                }
            }
        }
    }
}


template<size_t C, size_t H, size_t W, size_t stride>
void MaxPooling<C,H,W,stride>::forward(const BatchedTensor<float, C, H, W>& input, BatchedTensor<float, C, OUT_H, OUT_W>& output) {
    size_t batch_size = input.getDim(0);

    assert(H % stride == 0);
    assert(W % stride == 0);

    // size_t output_width = input_width / stride ,
    //     output_height = input_height / stride;

    //maximas = new Tensor<unsigned int, 4>({ batch_size, channels, input_height, input_width });
    // maximasX = new Tensor<unsigned int, 4>({ batch_size, channels, output_height, output_width });
    // maximasY = new Tensor<unsigned int, 4>({ batch_size, channels, output_height, output_width });
    if (maximasX.data == nullptr || maximasX.batch_size != batch_size || maximasY.batch_size != batch_size || maximasY.data == nullptr)
    {
        if (this->maximasX.data != nullptr) delete[] this->maximasX.data;
        if (this->maximasY.data != nullptr) delete[] this->maximasY.data;
        unsigned int *dataX = new unsigned int[batch_size * C * OUT_H * OUT_W];
        unsigned int *dataY = new unsigned int[batch_size * C * OUT_H * OUT_W];
        maximasX.data = dataX;
        maximasY.data = dataY;
        maximasX.batch_size = batch_size;
        maximasY.batch_size = batch_size;
    }

    // std::cout << input << std::endl;
    // Tensor<float, 4> output({ batch_size, channels, output_height, output_width });
    maximasX.set(0);
    maximasY.set(0);
    output.set(0);  
    maxpooling_forward_cpu(input, maximasX, maximasY, output);

}

template<size_t C, size_t H, size_t W, size_t stride>
void MaxPooling<C,H,W,stride>::backward(const BatchedTensor<float, C, OUT_H, OUT_W>& error, BatchedTensor<float, C, H, W>& error_output){
    size_t batch_size = error.getDim(0);
    error_output.set(0);

	maxpooling_backward_cpu (error, maximasX, maximasY, error_output);
    // std::cout << "MaxPooling backward" << std::endl;
    // std::cout << error << std::endl;
    // delete[] this->maximasX.data;
    // delete[] this->maximasY.data;
}

