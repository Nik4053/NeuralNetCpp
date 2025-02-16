/* Copyright (C) 2024 Niklas Heidenreich - All Rights Reserved */
#pragma once
#include "tensor.hpp"


namespace tel{
    template <typename T, std::size_t... DIMS>
    class BatchedTensor {
    private:
    public:
        T *data = nullptr;
        size_t batch_size;
        size_t size() const { return Tensor<T,DIMS...>::SIZE * batch_size; }
        size_t getDim(size_t i) const { if(i == 0) return batch_size; else return Tensor<T,DIMS...>::getDim(i-1); }
        
        BatchedTensor(size_t batch_size, T *data) : data(data), batch_size(batch_size) {}
        template <typename ...Args>
        size_t idx(size_t b, Args ... args) const { assert(b < batch_size); return b*Tensor<T,DIMS...>::SIZE + Tensor<T,DIMS...>::idx(args...); }
        template <typename ...Args>
        T operator()(size_t b, Args ... args) const { assert(b < batch_size); return data[b*Tensor<T,DIMS...>::SIZE + Tensor<T,DIMS...>::idx(args...)]; }
        template <typename ...Args>
        T& operator()(size_t b, Args ... args) {assert(b < batch_size); return data[b*Tensor<T,DIMS...>::SIZE + Tensor<T,DIMS...>::idx(args...)]; }
        template <typename ...Args>
        T operator()(size_t x) const { assert(x < size()); return data[x]; }
        template <typename ...Args>
        T& operator()(size_t x) { assert(x < size()); return data[x]; }
        
        typename internal::genTen<T,DIMS...>::type operator[](size_t b) const { assert(b < batch_size); typename internal::genTen<T,DIMS...>::type t(data+b*internal::genTen<T,DIMS...>::type::SIZE); return t; }
        // will only reshape the internal Tensor. batch_size will stay the same
        template <std::size_t ...NDIMS>
        BatchedTensor<T,NDIMS...> reshape(){static_assert(Tensor<T,DIMS...>::SIZE == Tensor<T,NDIMS...>::SIZE,"SIZE of reshaped Tensor has to stay the same!"); BatchedTensor<T,NDIMS...> t(batch_size,data); return t; }
        template <std::size_t ...NDIMS>
        BatchedTensor<T,NDIMS...> reshape(size_t new_batch_size){ assert(this->size == (new_batch_size * Tensor<T,NDIMS...>::SIZE)); BatchedTensor<T,NDIMS...> t(new_batch_size,data); return t; }
        // void print(int precision=5){
        //     std::cout <<std::fixed<<std::setprecision(precision)<< *this <<std::endl;
        // }

        BatchedTensor<T,Tensor<T,DIMS...>::SIZE> flatten() const {BatchedTensor<T,Tensor<T,DIMS...>::SIZE> t(batch_size,data); return t; }

        void set(const T& val){ for(size_t i=0; i < size(); i++) data[i] = val; }
        template <std::size_t SSIZE>
        void set(const Tensor<T,SSIZE>& other) { assert(size == SSIZE); for(size_t i=0; i < size(); i++) data[i] = other.data[i]; }
        void set(const BatchedTensor<T,DIMS...>& other) { assert(batch_size == other.batch_size); for(size_t i=0; i < size(); i++) data[i] = other.data[i]; }
        

        // math operations
        // max
        T max() const { return *std::max_element(data, data + size()); }

    };

    template <typename T, std::size_t... DIMS>
    std::ostream& operator<<(std::ostream& os, const BatchedTensor<T,DIMS...>& t)
    {
        os << "[";
        for(size_t d=0; d < t.batch_size; d++){
            if(d + 1 == t.batch_size) os <<  t[d] ;
            else  os <<  t[d] << ",";
        }
        os << "]";
        return os;
    }
} // namespace tel