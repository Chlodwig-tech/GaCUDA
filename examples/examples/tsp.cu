#include <iostream>
#include <cuda_runtime.h>

#include "../gacuda/gacuda.h"

__device__ int *costs;

template<int Size> class TSP : public Organism<int, int, Size>{
public:
    __device__ void print(){
        for(int i = 0; i < Size; i++){
            printf("%d ", this->genes[i]);
        }
        printf("-> %d\n", this->fvalue);
    }
    template<typename r> __device__ void random(curandState *state, r a, r b){
        for(int i = 0; i < Size; i++){
            this->genes[i] = i;
        }
        for(int i = Size - 1; i > 0; i--){
            int x = curand_uniform(state) * 100;
            int j = x % (i + 1);
            int temp = this->genes[i];
            this->genes[i] = this->genes[j];
            this->genes[j] = temp;
        }
    }
    __device__ void own_crossover(curandState *state, TSP<Size> *second_parent, TSP<Size> *child){
        int part = curand(state) % Size;
        for(int i = 0; i < part; i++){
            child->genes[i] = this->genes[i];
        }
        for(int i = 0; i < Size; i++){
            int gene = second_parent->genes[i];
            bool used = false;
            for(int j = 0; j < part && !used; j++){
                if(child->genes[j] == gene){
                    used = true;
                }
            }
            if(!used){
                child->genes[part] = gene;
                part++;
            }
        }
    }
    __device__ void fitness(){
        int sum = 0;
        for(int i = 0; i < Size - 1; i++){
            sum += costs[this->genes[i] * Size + this->genes[i + 1]];
        }
        sum += costs[this->genes[Size - 1] * Size + this->genes[0]];
        this->fvalue = sum;
    }
};


int main(){
    const int individual_size = 8;
    const int population_size = 8192;

    int prices[] = {
        0, 5, 7, 8, 7, 6, 6, 4, 
        5, 0, 8, 1, 8, 4, 7, 2, 
        7, 8, 0, 8, 2, 6, 7, 5, 
        8, 1, 8, 0, 5, 7, 8, 5, 
        7, 8, 2, 5, 0, 2, 2, 7, 
        6, 4, 6, 7, 2, 0, 3, 6, 
        6, 7, 7, 8, 2, 3, 0, 4, 
        4, 2, 5, 5, 7, 6, 4, 0
    };

    int *ah;
    CUDA_CALL(cudaMalloc((void **)&ah, individual_size * individual_size * sizeof(int)), "cudaMalloc ah");
    CUDA_CALL(cudaMemcpy(ah, prices, individual_size * individual_size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy ah <- prices");
    CUDA_CALL(cudaMemcpyToSymbol(costs, &ah, sizeof(int *), size_t(0),cudaMemcpyHostToDevice), "cudaMemcpyToSymbol costs");

    Population<TSP<individual_size>> p(population_size);
    p.set_current_best(9999);
    p.random(0, 0);
    p.fitness();
    p.printP(2);

    cudaDeviceSynchronize();
    
    int number_of_epochs = 10000;
    for(int i = 0; i < number_of_epochs; i++){
        p.mutate(MUTATION_INVERSION, 3.0f); // 3% for mutation
        p.crossover(CROSSOVER_OWN, 20.0f); // 30% of the best organisms
        p.sortAll(); // sort all organisms and make selection
    }
    
    cudaDeviceSynchronize();
    p.printP(2);
    p.print_current_best();
    CUDA_CALL(cudaFree(ah), "cudaFree ah");

    return 0;
}