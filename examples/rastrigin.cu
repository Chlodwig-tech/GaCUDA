#include <iostream>
#include <cuda_runtime.h>

#include "../gacuda/gacuda.h"

template<int Size> class Rastrigin : public Organism<float, float, Size>{
public:
    __device__ void print(){
        for(int i = 0; i < Size; i++){
            printf("%f ", this->genes[i]);
        }
        printf("-> %f\n", this->fvalue);
    }
    __device__ void fitness(){
        // f(x) = A * n + (\sum_{i=0}^{n} xi^2 - A*cos(2 * π * xi))
        float sum = 10 * Size;
        for (int i = 0; i < Size; ++i) {
            sum += (this->genes[i] * this->genes[i] - 10 * cos(2 * M_PI * this->genes[i]));
        }
        this->fvalue = sum;
    }
};


int main(){

    const int individual_size = 2; // individual size
    const int population_size = 8192; // population size

    Population<Rastrigin<individual_size>> p(population_size);
    p.random(-5.12f, 5.12f);
    p.set_current_best(9999.0f);
    p.fitness();

    p.printP(2);

    int number_of_epochs = 10000;
    for(int i = 0; i < number_of_epochs; i++){
        //p.mutate(MUTATION_OWN, 3.0f); // 3% for mutation
        p.shift_mutate(0.001f, 3.0f);
        p.crossover(CROSSOVER_UNIFORM, 20.0f); // 30% of the best organisms
        p.sortAll(); // sort all organisms and make selection
    }

    p.printP(2);
    p.print_current_best();
    cudaDeviceSynchronize();
}