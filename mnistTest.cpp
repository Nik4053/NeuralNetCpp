#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <vector>
#include "batchedtensor.hpp"
#include "dense.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include "crossentropyloss.hpp"
#include "conv.hpp"
#include "maxpooling.hpp"
#include "flatten.hpp"
#include "sigmoid.hpp"

using namespace std;

template< size_t IN_C, size_t filter_size, size_t H, size_t W, size_t LABEL>
class MnistConvNet {
	// const size_t input_channels = 1;
	// const size_t filter_size = 5;
	float learning_rate;
	const static size_t OUT_C1 = 10;
	OptimizerSGD<OUT_C1 * IN_C * filter_size * filter_size, OUT_C1> optimizerConv; // checked
	Conv2D<IN_C, OUT_C1, filter_size,H,W> c;
	ReLu< OUT_C1 *  H* W> relu; // sigmoid // checked
	const static size_t pool_stride = 2;
	const static size_t maxplool_out =  H / 2 * W / 2;
	MaxPooling<OUT_C1, H, W,pool_stride > maxpool; // checked
	const static size_t Dense_Out1 = 120;
	OptimizerSGD<OUT_C1 *maxplool_out*Dense_Out1,Dense_Out1> optimizer2; // checked
	Dense<OUT_C1 *maxplool_out, Dense_Out1> dense1; // checked
	// ReLuCPU<2> sigLayer; // checked
	Sigmoid<Dense_Out1> sigLayer; // checked
	ReLu<Dense_Out1> relu2; // sigmoid // checked
	OptimizerSGD<Dense_Out1 * LABEL, LABEL> optimizer2_2; // checked
	Dense<Dense_Out1 ,LABEL> dense2; // sigmoid // checked
	SoftMax<LABEL> sMaxLayer; // checked
	Flatten<OUT_C1, H / 2 , W / 2> flatten; // checked // TODO
	CrossEntropyLoss<LABEL> loss; // checked
public:
	MnistConvNet(float learning_rate):learning_rate(learning_rate), optimizerConv(learning_rate), c(&optimizerConv), optimizer2(learning_rate) , dense1(&optimizer2), optimizer2_2(learning_rate), dense2(&optimizer2_2) {
		c.initialize(0,1);
		dense1.initialize(0, 1);
		dense2.initialize(0, 1);
	}

	void forward(BatchedTensor<float, IN_C,H,W>& input, BatchedTensor<float, LABEL>& output) {
		int batch_size = input.batch_size;
		float* data1 = new float[batch_size*OUT_C1 * H * W];
		BatchedTensor<float, OUT_C1, H, W> c1(batch_size, data1);
		c1.set(0);
		c.forward(input, c1);
		BatchedTensor<float, OUT_C1 * H * W> r1(batch_size, data1);
		// relu.forward(r1, r1);
		float* data2 = new float[batch_size*OUT_C1 *maxplool_out];
		BatchedTensor<float, OUT_C1, H / 2 , W / 2> m1(batch_size, data2);
		// std::cout << "input: " << r1 << std::endl;
		maxpool.forward(c1, m1);
		auto m1f = flatten.forward(m1);
		float* data3 = new float[batch_size*Dense_Out1];
		BatchedTensor<float, Dense_Out1> d1(batch_size, data3);
		dense1.forward(m1f, d1);
		sigLayer.forward(d1, d1);
		float* data4 = new float[batch_size*LABEL];
		BatchedTensor<float, LABEL> d2(batch_size, data4);
		dense2.forward(d1, d2);
		sMaxLayer.forward(d2, output);

		delete[] data1;
		delete[] data2;
		delete[] data3;
		delete[] data4;
	}

	void backward(BatchedTensor<float, LABEL>& error) {
		sMaxLayer.backward(error);
		size_t batch_size = error.batch_size;
		float* data4 = new float[batch_size*Dense_Out1];
		BatchedTensor<float, Dense_Out1> d1(batch_size, data4);
		dense2.backward(error, d1);
		sigLayer.backward(d1);
		float* data3 = new float[batch_size*OUT_C1 *maxplool_out];
		BatchedTensor<float, OUT_C1 *maxplool_out> m1f(batch_size, data3);
		dense1.backward(d1, m1f);
		auto m1 = flatten.backward(m1f);
		float* data2 = new float[batch_size* OUT_C1 * H  * W];
		BatchedTensor<float, OUT_C1, H, W > m2(batch_size, data2);
		maxpool.backward(m1, m2);
		BatchedTensor<float, OUT_C1 * H * W> r1(batch_size, data2);
		// relu.backward(r1);
		float* data1 = new float[batch_size* 1 * H * W];
		BatchedTensor<float, 1, H, W> c1(batch_size, data1);
		c.backward(m2, c1);

		delete[] data1;
		delete[] data2;
		delete[] data3;
		delete[] data4;
	}

	float train(BatchedTensor<float, IN_C, H, W>& input, BatchedTensor<float, LABEL>& labels) {
		static int ctr = 0;
		ctr++;
		size_t batch_size = input.batch_size;
		float* data = new float[batch_size*LABEL];
		BatchedTensor<float, LABEL> output(batch_size, data);
		output.set(0);
		forward(input, output);
		float l = loss.forward(output, labels);
		loss.backward(labels);
		// std::cout << "forward: " << labels << std::endl;
		backward(labels);
		// std::cout << "forward: " << output << std::endl;

		delete[] data;
		std::cout << "TrainLoss: " << l << std::endl;
		return l;
	}

	float test(BatchedTensor<float, IN_C, H, W>& input, BatchedTensor<float, LABEL>& labels) {
		size_t batch_size = input.batch_size;
		float* data = new float[batch_size*LABEL];
		BatchedTensor<float, LABEL> output(batch_size, data);
		output.set(0);
		forward(input, output);
		float l = loss.forward(output, labels);
		std::cout << "TestLoss: " << l << std::endl;
		size_t correct = 0;
		for (size_t b = 0; b < batch_size; b++)
		{
			float* maxElement = std::max_element(&output(b, 0), &output(b, 0) + output.getDim(1));
			size_t maxIdx = std::distance(&output(b, 0), maxElement);
			if (labels(b, maxIdx) == 1) correct++;
		}
		size_t wrong = batch_size - correct;
		float accuracy = (double)correct / (correct + wrong);
		std::cout << "Accuracy: " << accuracy << " -> " << correct << "/" << batch_size << std::endl;
		delete[] data;
		return l;
	}

    // Tensor<float, 2> forward(const Tensor<float, 4>& input) {
	// 	Tensor<float, 4> c1 = c.forward(input);
	// 	// cout << "c1: " << c1.sum() << endl;
	// 	Tensor<float, 4> r1 = relu.forward(c1);
	// 	// cout << "r1: " << r1.sum() << endl;
	// 	Tensor<float, 4> m1 = maxpool.forward(c1);
	// 	// cout << "m1: " << m1.sum() << endl;
	// 	Tensor<float, 2> f1 = flatten.forward(m1);
	// 	// cout << "f1: " << f1.sum() << endl;
	// 	Tensor<float, 2> d1 = dense1.forward(f1);
	// 	// cout << "d1: " << d1.sum() << endl;
	// 	Tensor<float, 2> s1 = sigLayer.forward(d1);
	// 	// cout << "s1: " << s1.sum() << endl;
	// 	Tensor<float, 2> d2 = dense2.forward(s1);
	// 	// cout << "d2: " << d2.sum() << endl;
	// 	Tensor<float, 2> so1 = sMaxLayer.forward(d2);
	// 	// cout << "so1: " << so1.sum() << endl;
	// 	// cout << so1.getData() << endl;
	// 	return so1;
	// }

	// Tensor<float, 4> backward(Tensor<float, 2>& error) {
	// 	Tensor<float, 2> so1 = sMaxLayer.backward(error);
	// 	Tensor<float, 2> d2 = dense2.backward(so1);
	// 	Tensor<float, 2> s1 = sigLayer.backward(d2);
	// 	Tensor<float, 2> d1 = dense1.backward(s1);
	// 	Tensor<float, 4> f1 = flatten.backward(d1);
	// 	Tensor<float, 4> m1 = maxpool.backward(f1);
	// 	 Tensor<float, 4> r1 = relu.backward(m1);
	// 	Tensor<float, 4> c1 = c.backward(m1);
	// 	return c1;
	// }

	// void train(Tensor<float, 4>& input, Tensor<float, 2>& labels) {
	// 	Tensor<float, 2> fw = forward(input);
	// 	// fw.print2D();

	// 	float l = loss.forward(fw, labels);
	// 	cout << "loss= " << l << endl;
	// 	Tensor<float, 2> lb = loss.backward(labels);
	// 	// lb.print2D();
	// 	Tensor<float, 4> back = backward(lb);
	// 	// back.print();
	// }

	// void test(Tensor<float, 4>& input, Tensor<float, 2>& labels) {
	// 	Tensor<float, 2> fw = forward(input);
	// 	float l = loss.forward(fw, labels);
	// 	// set max to 1 nd other to 0
	// 	for (size_t b = 0; b < fw.getDim(0); b++)
	// 	{
	// 		float* maxElement = std::max_element(&fw(b, 0), &fw(b, 0) + fw.getDim(1));
	// 		size_t maxIdx = std::distance(&fw(b, 0), maxElement);
	// 		for (size_t i = 0; i < fw.getDim(1); i++) {
	// 			fw(b, i) = 0;
	// 		}
	// 		fw(b, maxIdx) = 1;
	// 	}
	// 	size_t correct = 0;
	// 	for (size_t b = 0; b < fw.getDim(0); b++)
	// 	{
	// 		size_t tmp = 0;
	// 		for (size_t i = 0; i < fw.getDim(1); i++) {
	// 			if (fw(b, i) == labels(b, i)) tmp++;
	// 		}
	// 		if (tmp == fw.getDim(1)) correct++;
	// 	}
	// 	size_t wrong = labels.getDim(0) - correct;
	// 	float accuracy = (double)correct / (correct + wrong);
	// 	std::cout << "Accuracy: " << accuracy << " -> " << correct << "/" << labels.getDim(0) << std::endl;
	// }

};


template< class T >
class DataLoader {
private:
	size_t ctr = 0;
	size_t batch_size;
	size_t num_categories;
	std::vector<std::vector<T>> train_data;
	std::vector<float> train_label;
	std::vector<std::vector<T>> test_data;
	std::vector<float> test_label;
	// https://stackoverflow.com/a/1267878
	template< class TT >
	void reorder(vector<TT>& v, vector<size_t> const& order) {
		for (int s = 1, d; s < order.size(); ++s) {
			for (d = order[s]; d < s; d = order[d]);
			if (d == s) while (d = order[d], d != s) swap(v[s], v[d]);
		}
	}
	void shuffleData(std::vector<std::vector<T>>& data, std::vector<float>& label) {
		assert(data.size() == label.size());
		size_t datasize = data.size();
		std::vector<size_t> indexes(datasize);
		std::iota(indexes.begin(), indexes.end(), 0); // cpp11
		std::random_shuffle(indexes.begin(), indexes.end());
		reorder(data, indexes);
		reorder(label, indexes);
	}
	void shuffleData(std::vector<std::vector<T>>& data, std::vector<std::vector<float>>& label) {
		assert(data.size() == label.size());
		size_t datasize = data.size();
		std::vector<size_t> indexes(datasize);
		std::iota(indexes.begin(), indexes.end(), 0); // cpp11
		std::random_shuffle(indexes.begin(), indexes.end());
		reorder(data, indexes);
		reorder(label, indexes);
	}
	std::vector<std::vector<float>> oneHotEncode(std::vector<float>& label, size_t num_categories) {
		size_t data_size = label.size();
		std::vector<std::vector<float>> encoded_label;//{ data_size, num_categories }, 0);
		encoded_label.reserve(data_size);
		for (size_t b = 0; b < data_size; b++)
		{
			assert(label[b] < num_categories);
			std::vector<float> one_hot(num_categories,0);
			one_hot[label[b]] = 1;
			encoded_label.push_back(one_hot);
		}
		return encoded_label;
	}

	void splitData(std::vector<std::vector<T>>& data, std::vector<float>& label, float test_split) {
		assert(data.size() == label.size());
		size_t data_size = data.size();
		size_t input_size = data[0].size();
		size_t test_size = test_split * data_size;
		size_t train_size = data_size - test_size;
		train_data.reserve(train_size);
		train_label.reserve(train_size);
		test_data.reserve(test_size);
		test_label.reserve(test_size);
		train_data.assign(data.begin(), data.begin() + train_size);
		train_label.assign(label.begin(), label.begin() + train_size);
		test_data.assign(data.begin() + train_size, data.end());
		test_label.assign(label.begin() + train_size, label.end());
	}
	void nextEpoch() {
		ctr = 0;
		// shuffleData(train_data, train_label);
	}
public:
	DataLoader(std::vector<std::vector<T>> &data, std::vector<float> &label, size_t num_categories, size_t batch_size, float test_split = 0.3):batch_size(batch_size), num_categories(num_categories){
		assert(data.size() == label.size());
		assert(batch_size < data.size());
		assert(test_split < 1);
		// shuffleData(data, label);
		splitData(data, label, test_split);
	}
	
	DataLoader(std::vector<std::vector<T>>& train_data, std::vector<float>& train_label, std::vector<std::vector<T>>& test_data, std::vector<float>& test_label, size_t num_categories, size_t batch_size) 
		:batch_size(batch_size), num_categories(num_categories), train_data(train_data), train_label(train_label), test_data(test_data), test_label(test_label)
	{
		nextEpoch();
	}

	std::pair< std::vector<std::vector<T>>, std::vector<std::vector<float>>>  nextBatch() {
		std::vector<std::vector<T>> batch_data;
		std::vector<float> batch_label;
		batch_data.reserve(batch_size);
		batch_label.reserve(batch_size);
		if (ctr + batch_size >= train_data.size()) nextEpoch();
		batch_data.assign(train_data.begin() + ctr, train_data.begin() + ctr + batch_size);
		batch_label.assign(train_label.begin() + ctr, train_label.begin() + ctr + batch_size);
		ctr += batch_size;
		std::pair< std::vector<std::vector<T>>, std::vector<std::vector<float>>> PAIR;
		PAIR.first = batch_data;
		PAIR.second = oneHotEncode(batch_label, num_categories);
		return PAIR;
	}

	std::pair< std::vector<std::vector<T>>, std::vector<std::vector<float>>>  testData() {

		std::pair< std::vector<std::vector<T>>, std::vector<std::vector<float>>> PAIR;
		PAIR.first = test_data;
		PAIR.second = oneHotEncode(test_label, num_categories);
		return PAIR;
	}

};



template <size_t train_size, size_t widthheight>
void read_train_data(Tensor<float, train_size, widthheight>& data_train, Tensor<float, train_size>& label_train) {
	std::ifstream csvread;
	csvread.open("mnist_train.csv", ios::in);
	if (csvread) {
		//Datei bis Ende einlesen und bei ';' strings trennen
		string s;
		int data_pt = 0;
		while (getline(csvread, s)) {
			stringstream ss(s);
			int pxl = 0;
			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				if (pxl == 0) {
					label_train(data_pt) = stoi(substr);
				}
				else {
					data_train(data_pt, pxl - 1) = stoi(substr);
				}
				pxl++;
			}
			data_pt++;
		}
		csvread.close();
	}
	else {
		//cerr << "Fehler beim Lesen!" << endl;
		cerr << "Can not read data!" << endl;
	}
}
template <size_t test_size, size_t widthheight>
void read_test_data(Tensor<float, test_size, widthheight >& data_test, Tensor<float, test_size>& label_test) {
	ifstream csvread;
	csvread.open("mnist_test.csv", ios::in);
	if (csvread) {
        std::cout << "Reading in test data" << std::endl;
		//Datei bis Ende einlesen und bei ';' strings trennen
		string s;
		int data_pt = 0;
		while (getline(csvread, s)) {
			stringstream ss(s);
			int pxl = 0;
			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				if (pxl == 0) {
					label_test(data_pt) = stoi(substr);
                
				}
				else {
					data_test(data_pt, pxl - 1) = stoi(substr);
				}
				pxl++;
			}
			data_pt++;
		}
		csvread.close();
	}
	else {
		//cerr << "Fehler beim Lesen!" << endl;
		cerr << "Can not read data!" << endl;
	}
}


template <size_t size, size_t channels, size_t width, size_t height>
void extractImages(Tensor<float, size, width * height>& data, Tensor<float, size,channels,height,width>& images) {
	// assert(data.getDim(1) == images.getDim(2) * images.getDim(3));
	// assert(data.getDim(0) == images.getDim(0));
	// assert(images.getDim(1) == 1);
	// size_t height = images.getDim(2);
	// size_t width = images.getDim(3);
	for (size_t i = 0; i < images.getDim(0); i++) {
		for (size_t k = 0; k < images.getDim(2); k++) {
			for (size_t l = 0; l < images.getDim(3); l++) {
				images(i, 0, k, l) = data(i, k * height + l);
			}
		}
	}
}
template <size_t test_size, size_t widthheight>
vector<vector<vector<float>>> extractImagesToVec(Tensor<float, test_size, widthheight>& data) {
	vector<vector<vector<float>>> imgs;
	imgs.reserve(data.getDim(0));
	size_t height = 28;
	size_t width = 28;
	for (size_t i = 0; i < data.getDim(0); i++) {
		vector<vector<float>> img;
		img.reserve(height);
		for (size_t k = 0; k < height; k++) {
			vector<float> img_line(width);
			for (size_t l = 0; l < width; l++) {
				img_line[l] = data(i, k * height + l);
			}
			img.push_back(img_line);
		}
		imgs.push_back(img);
	}
	return imgs;
}

template <size_t SIZE>
vector<float> toVec(Tensor<float, SIZE> &t) {
	vector<float> vec(t.SIZE);
	for (size_t i = 0; i < t.SIZE; i++)
	{
		vec[i] = t(i);
	}return vec;
}

template <size_t SIZE>
void toTensor(std::vector<std::vector<float>> &data, BatchedTensor<float, SIZE> t) {
	size_t batch_size = data.size();
	size_t data_size = data[0].size();
    assert(data_size == SIZE);
	for (size_t b = 0; b < batch_size; b++)
	{
		for (size_t i = 0; i < data_size; i++)
		{
			t(b, i) = data[b][i];
		}
	}
}
template <size_t H, size_t W>
void toTensor(std::vector<std::vector<std::vector<float>>>& data, BatchedTensor<float,1, H, W> t) {
	size_t batch_size = data.size();
	size_t height = data[0].size();
	size_t width = data[0][0].size();
	assert(height == H);
	assert(width == W);
	for (size_t b = 0; b < batch_size; b++)
	{
		for (size_t h = 0; h < height; h++)
		{
			for (size_t w = 0; w < width; w++)
			{
				t(b, 0, h, w) = data[b][h][w];
			}
		}
	}
}

int main() {
    // load data
	const size_t width = 28;
	const size_t height = 28;
	const size_t channels = 1;
	const size_t num_classes = 10;
	const size_t train_size = 60000;
	const size_t test_size = 10000;
	// float* t = new float[200000000L];
	// cout << t[12345] << endl;
	const size_t epochs = 30;
	const size_t batch_size = 50;
	const int filter_size = 10;
	const float learning_rate = 0.001;
	const size_t input_channels = 1;
	MnistConvNet< input_channels, filter_size,height,width,num_classes> nn(learning_rate);

	// float *data_in = new float[batch_size * channels * height * width];
	// float *label_in = new float[batch_size * num_classes];
	// BatchedTensor<float, channels, height, width> data(batch_size,data_in);
	// BatchedTensor<float, num_classes> label(batch_size,label_in);
	// data.set(0);
	// label.set(0);
	// nn.train(data, label);

	// delete[] data_in;
	// delete[] label_in;
	// return 1;

    float *data_train_array = new float[train_size * width * height];
    float *label_train_array = new float[train_size];
    float *data_test_array = new float[test_size *width * height];
    float *label_test_array = new float[test_size];

	Tensor<float, train_size, width * height > data_train(data_train_array);
	Tensor<float, test_size, width * height > data_test(data_test_array);
	Tensor<float, train_size> label_train(label_train_array);
	Tensor<float, test_size> label_test(label_test_array);
	std::cout << "Reading in test data" << std::endl;
	read_test_data(data_test, label_test);
	std::cout << "Reading in train data" << std::endl;
	read_train_data(data_train, label_train);

    float *data_train_imgs_array = new float[train_size * channels * height * width];
    float *data_test_imgs_array= new float[test_size * channels * height * width];

	// Tensor<float,  train_size,channels,height,width> data_train_imgs(data_train_imgs_array);
	// Tensor<float, test_size,channels,height,width> data_test_imgs(data_test_imgs_array);
	// std::cout << "Extracting data" << std::endl;
	// extractImages(data_train, data_train_imgs);
	// extractImages(data_test, data_test_imgs);

	std::cout <<  "data_train_imgs: " << std::accumulate(data_train.data, data_train.data + 1000, 0.0) << std::endl;
	// 255 -> 1
	// data_train.set(data_train/TVal<Tensor<float, train_size, width * height >::SIZE>(255));
	data_train /= 255;
	data_test /= 255;
	// for (size_t i = 0; i < data_train.size(); i++)
	// {
	// 	data_train.data[i] /= 255.0f;
	// }
	// for (size_t i = 0; i < data_test.size(); i++)
	// {
	// 	data_test.flatten()(i) /= 255;
	// }
	std::cout <<  "data_train_imgs: " << std::accumulate(data_train.data, data_train.data + 1000, 0.0) << std::endl;


	vector<vector<vector<float>>> data_train_imgs_vec = extractImagesToVec(data_train);
	vector<vector<vector<float>>> data_test_imgs_vec = extractImagesToVec(data_test);

	//DataLoader< vector<float>> loader(data_train_imgs_vec, toVec(label_train), data_test_imgs_vec, toVec(label_test), num_classes, batch_size);
	//DataLoader< vector<float>> loader(data_train_imgs_vec, toVec(label_train), data_test_imgs_vec, toVec(label_test), num_classes, batch_size);
	vector<float> label_train_vec = toVec(label_train);
	vector<float> label_test_vec = toVec(label_test);
	// std::cout << "label_train: " << std::accumulate(label_train.data, label_train.data + 1000, 0.0) << std::endl;
	std::cout << "data_train_imgs_vec: " << std::accumulate(data_train_imgs_vec[0][0].begin(), data_train_imgs_vec[0][0].begin() + 32, 0.0) << std::endl;
	DataLoader< vector<float>> loader(data_train_imgs_vec, label_train_vec, data_test_imgs_vec, label_test_vec, num_classes, batch_size);
	for (size_t i = 0; i < epochs; i++)
	{
		if (i % 10 == 0) cout << "iter: " << i << endl;
		auto batch = loader.nextBatch();
        assert( batch.first.size() == batch_size);
        assert( batch.second.size() == batch_size);
        float *batch_data_array = new float[batch_size * channels * height * width];
        float *batch_label_array = new float[batch_size * num_classes];
        BatchedTensor<float, channels,height,width> batch_data(batch_size,batch_data_array);
        BatchedTensor<float, num_classes> batch_label(batch_size,batch_label_array);
		// batch_data.set(0);
		// batch_label.set(0);
		toTensor(batch.first, batch_data);
		toTensor(batch.second, batch_label);
		nn.train(batch_data, batch_label);
		delete[] batch_data_array;
		delete[] batch_label_array;
		// break;
	}
	auto test = loader.testData();
    assert( test.first.size() == test_size);
    assert( test.second.size() == test_size);
    float *test_data_array = new float[test_size * channels * height * width];
    float *test_label_array = new float[test_size * num_classes];
    BatchedTensor<float, channels,height,width> test_data(test_size,test_data_array);
    BatchedTensor<float, num_classes> test_label(test_size,test_label_array);
    toTensor(test.first, test_data);
    toTensor(test.second, test_label);
	nn.test(test_data, test_label);

	delete[] data_train_array;
	delete[] label_train_array;

	delete[] data_test_array;
	delete[] label_test_array;

	delete[] data_train_imgs_array;
	delete[] data_test_imgs_array;

	delete[] test_data_array;
	delete[] test_label_array;



}
