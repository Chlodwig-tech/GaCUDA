#include <iostream>
#include <cuda_runtime.h>

#include "gacuda/gacuda.h"

template<int Size> class Sum : public Organism<float, float, Size>{
public:
    __device__ void print(){
        for(int i = 0; i < Size; i++){
            printf("%f ", this->genes[i]);
        }
        printf("-> %f\n", this->fvalue);
    }
    __device__ void fitness(){
        float sum = 0;
        for(int i = 0; i < Size; i++){
            sum += this->genes[i];
        }
        this->fvalue = sum;
    }
};


int main(){

    Population<Sum<4>> p1(4);
    p1.bplogspace(2.0f, 3.0f, 10.0f);
    p1.fitness();
    
    p1.printP();

    p1.plogspace(2.0f, 3.0f, 10.0f);
    p1.fitness();
    p1.printP();

}