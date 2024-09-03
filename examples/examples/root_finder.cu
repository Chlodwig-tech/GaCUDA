#include <iostream>
#include <cuda_runtime.h>

#include "../gacuda/gacuda.h"

template<int Size> class RootFinder : public Organism<float, float, Size>{
public:
    __device__ void print(){
        for(int i = 0; i < Size; i++){
            printf("%f ", this->genes[i]);
        }
        printf("-> %f\n", this->fvalue);
    }
    __device__ void fitness(){
        // f(x0, x1, x2, x3, x4, x5, x6, x7) =
        // = x0^7 + 3x1^6 - x2^5 + 2.4x3^4 - 1.02x4^3 - x5 +12x6^2 - x7
        float f = 0;
        f += pow(this->genes[0], 7);
        f += 3 * pow(this->genes[1], 6);
        f += pow(this->genes[2], 5);
        f += 2.4 * pow(this->genes[3], 4);
        f += 1.02 * pow(this->genes[4], 3);
        f += -this->genes[5];
        f += 12 * pow(this->genes[6], 2);
        f += -this->genes[7];

        this->fvalue = abs(f);
    }
};


int main(){

    const int individual_size = 8; // individual size
    const int population_size = 8192 * 8; // population size

    Population<RootFinder<individual_size>> p(population_size);
    p.random(-10.0f, 10.0f);
    p.set_current_best(9999.0f);
    p.fitness();

    p.printP(2);

    int number_of_epochs = 100000;
    for(int i = 0; i < number_of_epochs; i++){
        if(i % 10000 == 0){
            float best = p.get_best_value();
            printf("Epoch %d, current best: %f\n", i, best);
        }
        p.mutate(MUTATION_SCRAMBLE, 3.0f);
        p.crossover(CROSSOVER_UNIFORM, 20.0f); // 20% of the best organisms
        p.sortAll(); // sort all organisms and make selection
    }

    p.printP(2);
    p.print_current_best();
    cudaDeviceSynchronize();
}