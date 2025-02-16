#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>
#include "batchedtensor.hpp"
#include "dense.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include "crossentropyloss.hpp"


using namespace tel;


template <std::size_t IN_SIZE, std::size_t L1_SIZE, std::size_t L2_SIZE>
class IrisModel{
    public:
        float learning_rate;

        OptimizerSGD<IN_SIZE*L1_SIZE,L1_SIZE> optimizer1 = OptimizerSGD<IN_SIZE*L1_SIZE,L1_SIZE>(learning_rate);
        OptimizerSGD<L1_SIZE*L2_SIZE,L2_SIZE> optimizer2 = OptimizerSGD<L1_SIZE*L2_SIZE,L2_SIZE>(learning_rate);

        Dense<IN_SIZE,L1_SIZE> dense1 = Dense<IN_SIZE,L1_SIZE>(&optimizer1);
        Dense<L1_SIZE,L2_SIZE> dense2 = Dense<L1_SIZE,L2_SIZE>(&optimizer2);

        ReLu<L1_SIZE> relu = ReLu<L1_SIZE>();
        SoftMax<L2_SIZE> softmax = SoftMax<L2_SIZE>();

        CrossEntropyLoss<L2_SIZE> loss = CrossEntropyLoss<L2_SIZE>();

        IrisModel(float learning_rate):learning_rate(learning_rate){
            dense1.initialize(-1,1);
            dense2.initialize(-1,1);
        }

        void forward(const BatchedTensor<float,IN_SIZE> &input, BatchedTensor<float,L2_SIZE> &output){
            float* data1 = new float[input.batch_size*L1_SIZE];
            BatchedTensor<float,L1_SIZE> output1 = BatchedTensor<float,L1_SIZE>(input.batch_size, data1);
            dense1.forward(input, output1);
            relu.forward(output1, output1);
            dense2.forward(output1, output);
            delete[] data1;
            softmax.forward(output, output);
        }

        void backward(BatchedTensor<float,L2_SIZE> &error){
            size_t batch_size = error.batch_size;
            softmax.backward(error);
           
            float* data2 = new float[batch_size*L1_SIZE];
            BatchedTensor<float,L1_SIZE> error2 = BatchedTensor<float,L1_SIZE>(batch_size, data2);
            dense2.backward(error, error2);
            relu.backward(error2);
            float* data1 = new float[batch_size*IN_SIZE];
            BatchedTensor<float,IN_SIZE> error1 = BatchedTensor<float,IN_SIZE>(batch_size, data1);
            dense1.backward(error2, error1);
            
            delete[] data2;
            delete[] data1;
        }

        float train(BatchedTensor<float,4> &input, BatchedTensor<float,3> &target){
            float* data = new float[input.batch_size*L2_SIZE];
            BatchedTensor<float,L2_SIZE> output = BatchedTensor<float,L2_SIZE>(input.batch_size, data);
            output.set(0);
            forward(input, output);
            // output.print();
            
            float l = loss.forward(output, target);
            
            // std::cout << "loss: " << l << std::endl;
            delete[] data;

            loss.backward(target);
            // target.print();

            backward(target);
            return l;


        }

        float test(BatchedTensor<float,4> &input, BatchedTensor<float,3> &target){
            float* data = new float[input.batch_size*L2_SIZE];
            BatchedTensor<float,L2_SIZE> output = BatchedTensor<float,L2_SIZE>(input.batch_size, data);
            output.set(0);
            forward(input, output);
            float l = loss.forward(output, target);
            std::cout << "loss: " << l << std::endl;

            // set max to 1 and others to 0
            for(size_t i = 0; i < target.batch_size; i++){
                float max = output[i].max();
                for(size_t j = 0; j < output[i].SIZE; j++){
                    if(output[i][j] == max){
                        output[i][j] = 1;
                    } else {
                        output[i][j] = 0;
                    }
                }
            }

            // compare
            size_t correct = 0;
            for(size_t i = 0; i < target.batch_size; i++){
                if(output[i] == target[i]){
                    correct++;
                }
            }
            std::cout << "correct: " << correct <<"/" << target.batch_size<< std::endl;

            delete[] data;
            return l;
        }
        
    private:
};

// TODO: delete copy, move, assign constructors
class DataLoader{
    public:
        DataLoader(BatchedTensor<float,5> data, size_t batch_size){
            // this->data = data;
            size_t data_size = data.batch_size;
            this->batch_size = batch_size;
            this->current_index = 0;
            
            //shuffle 
            float *shuffled_data = new float[data.size()];
            BatchedTensor<float,5> shuffled(data_size, shuffled_data);
            shuffle(data, shuffled);
            //split into input and target
            float *input_data = new float[data_size*4];
            float *target_data = new float[data_size];
            BatchedTensor<float,4> input = BatchedTensor<float,4>(data_size, input_data);
            BatchedTensor<float,1> target = BatchedTensor<float,1>(data_size, target_data);
            split(shuffled, input, target);


            //split into train and test
            int test_size = data_size * 0.3;
            int train_size = data_size - test_size;
            std::cout << "data_size: " << data_size << std::endl;
            std::cout << "train_size: " << train_size << std::endl;
            std::cout << "test_size: " << test_size << std::endl;
            float *train_data = new float[train_size*4];
            float *train_target = new float[train_size*3];
            float *test_data = new float[test_size*4];
            float *test_target = new float[test_size*3];
            this->train_data = new BatchedTensor<float,4>(train_size, train_data);
            this->train_target = new BatchedTensor<float,3>(train_size, train_target); // one hot encoded
            this->test_data =  new BatchedTensor<float,4>(test_size, test_data);
            this->test_target =  new BatchedTensor<float,3>(test_size, test_target); // one hot encoded

            this->train_target->set(0);
            this->test_target->set(0);

            //fill train and test data
            for(size_t i = 0; i < test_size; i++){
                this->test_data->operator[](i).set(input[i]);
                this->test_target->operator[](i)[target[i][0]] = 1; // one hot encoded   
            }
            for(size_t i = test_size; i < data_size; i++){
                this->train_data->operator[](i - test_size).set(input[i]);
                this->train_target->operator[](i - test_size)[target[i][0]] = 1; // one hot encoded
            }

            // free memory
            delete[] shuffled_data;
            delete[] input_data;    
            delete[] target_data;

        }
        ~DataLoader(){
            delete[] train_data->data;
            delete[] train_target->data;
            delete[] test_data->data;
            delete[] test_target->data;
            delete train_data;
            delete train_target;
            delete test_data;
            delete test_target;
        }
        bool hasNext(){
            return current_index < train_data->batch_size;
        }
        void next(BatchedTensor<float,4> input, BatchedTensor<float,3> target){
            assert(hasNext());
            assert(current_index < train_data->batch_size);
            assert(input.batch_size == batch_size);
            assert(target.batch_size == batch_size);
            size_t start = current_index;
            size_t end = std::min(current_index + batch_size, train_data->batch_size);
            current_index = end;
            for(size_t i = 0; i < end - start; i++){
                input[i].set(train_data->operator[](i + start));
                target[i].set(train_target->operator[](i+start));
            }
        }
        void nextEpoch(){
            current_index = 0;
            epoch++;
            //shuffle
            // TODO: shuffle
        }

        BatchedTensor<float,4> *test_data;
        BatchedTensor<float,3> *test_target; // one hot encoded
    private:
        BatchedTensor<float,4> *train_data;
        BatchedTensor<float,3> *train_target; // one hot encoded

        size_t batch_size;
        size_t current_index;
        size_t epoch = 0;

        void shuffle(BatchedTensor<float,5> &dataToShuffle,  BatchedTensor<float,5> &shuffled){
            size_t data_size = dataToShuffle.batch_size;
            std::vector<size_t> indexes(data_size);
            std::iota(indexes.begin(), indexes.end(), 0); // cpp11
            std::random_shuffle(indexes.begin(), indexes.end());
            for(size_t i = 0; i < data_size; i++){
                shuffled[i].set(dataToShuffle[indexes[i]]);
            }
        }

        void split(BatchedTensor<float,5> dataToSplit, BatchedTensor<float,4> input, BatchedTensor<float,1> target){
            size_t data_size = dataToSplit.batch_size;
            for(size_t i = 0; i < data_size; i++){
                for(size_t j = 0; j < 4; j++){
                    input(i,j) = dataToSplit(i,j);
                }
                target(i,0) = dataToSplit(i,4);
            }
        }
};

/*
@params
    filename: path to file
    data: tensor to store data. sepal_length, sepal_width, petal_length, petal_width, label
*/
size_t readIrisData(const char* filename, BatchedTensor<float,5> data){
    //TODO: read data from file
    std::ifstream file(filename);
    std::string line;
    int i = 0;
    while (std::getline(file, line)){
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        for(int j = 0; j < 4; j++){
            std::getline(iss, token, ',');
            data(i,j) = std::stof(token);
        }
        std::getline(iss, token, ',');
        if(token == "Iris-setosa"){
            data(i,4) = 0;
        } else if(token == "Iris-versicolor"){
            data(i,4) = 1;
        } else if(token == "Iris-virginica"){
            data(i,4) = 2;
        } else {
            std::cout << "Error parsing label" << std::endl;
        }

        i++;
    }
    return i;

}

int main(){
    size_t epochs = 1000;
    size_t batch_size = 20;
    float learning_rate = 0.001;

    const size_t categories = 3;
    const size_t input_size = 4;

    // Read data
    const int n = 150;
    float *data_arr = new float[n*5];
    BatchedTensor<float,5> data(n,data_arr);
    if(readIrisData("data/iris.data", data) != n){
        std::cout << "Error reading data" << std::endl;
        return 1;
    }
    
    // Create model
    IrisModel<input_size,30,categories> model(learning_rate);
    // Use Dataloader
    DataLoader dataloader(data, batch_size);
    // Create input and target
    float *input_data  = new float[batch_size*input_size];
    float *target_data = new float[batch_size*categories];
    BatchedTensor<float,input_size> input  = BatchedTensor<float,4>(batch_size, input_data);
    BatchedTensor<float,categories> target = BatchedTensor<float,3>(batch_size, target_data); // one hot encoded
    input.set(0);
    target.set(0);
    std::vector<float> test_loss(epochs);
    std::vector<float> train_loss;
    // Train
    for(size_t epoch = 0; epoch < epochs; epoch++){
        float loss = 0;
        while(dataloader.hasNext()){
            dataloader.next(input, target);
            loss = model.train(input, target);
        }
        train_loss.push_back(loss);
        loss = model.test(*dataloader.test_data, *dataloader.test_target);
        test_loss[epoch] = loss;
        dataloader.nextEpoch();
    }
    // write loss to gnu plot file
    std::ofstream loss_file("loss.dat");
    for(size_t i = 0; i < train_loss.size(); i++){
        loss_file << i << " " << train_loss[i]<< " " << test_loss[i] << std::endl;
    }
    loss_file.close();

    
    // Free memory
    delete[] data_arr;
    delete[] input_data;
    delete[] target_data;
    return 0;


}