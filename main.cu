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

    IslandDeme<Sum<5>, 4> deme(8);
    deme.random(0.0f, 10.0f);
    deme.fitness();
    deme.printP();

    MUTATION mutations[] = {
        MUTATION_INVERSION,
        MUTATION_SWAP,
        MUTATION_INVERSION,
        MUTATION_SWAP
    };
    deme.deme_mutate(mutations, 100.0f);
    deme.printP();

    CROSSOVER crossovers[] = {
        CROSSOVER_ARITHMETIC,
        CROSSOVER_TWO_POINT,
        CROSSOVER_ARITHMETIC,
        CROSSOVER_TWO_POINT
    };
    deme.deme_crossover(crossovers, 100.0f);
    deme.deme_printP();

    deme.print_childrenP();

    cudaDeviceSynchronize();
    printf("\n\n\n\n\n\n");

    deme.deme_sort();
    deme.deme_printP();
    deme.print_childrenP();

    cudaDeviceSynchronize();
    printf("\n\n\n\n\n\nlol\n");
    deme.deme_printP();
    deme.deme_migrate();
    deme.deme_printP();

}