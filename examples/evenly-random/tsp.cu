#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#include "../../gacuda/gacuda.h"

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
    }
    template<typename r> __device__ void random(curandState *state, r a, r b){
        for(int i = 0; i < Size; i++){
            this->genes[i] = i;
        }
        if(a != b && a != 0){
            for(int i = Size - 1; i > 0; i--){
                int x = curand_uniform(state) * 100;
                int j = x % (i + 1);
                int temp = this->genes[i];
                this->genes[i] = this->genes[j];
                this->genes[j] = temp;
            }
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


int main(int argc, char *argv[]){
    int good_solution = 8000;

    const int individual_size = 13;
    const int population_size = std::atoi(argv[1]);
    float mutation_probability = std::atoi(argv[2]);
    float crossover_probability = std::atoi(argv[3]);

    std::stringstream filename;
    filename << "results/tsp/output-" << population_size << "-" << mutation_probability << "-" << crossover_probability << ".txt";
    std::ofstream outFile(filename.str());


    int prices[] = {
        0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972,
        2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579,
        713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260,
        1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987,
        1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371,
        1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999,
        2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701,
        213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099,
        2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600,
        875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162,
        1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200,
        2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504,
        1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0
    };

    int *ah;
    CUDA_CALL(cudaMalloc((void **)&ah, individual_size * individual_size * sizeof(int)), "cudaMalloc ah");
    CUDA_CALL(cudaMemcpy(ah, prices, individual_size * individual_size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy ah <- prices");
    CUDA_CALL(cudaMemcpyToSymbol(costs, &ah, sizeof(int *), size_t(0),cudaMemcpyHostToDevice), "cudaMemcpyToSymbol costs");

    Population<TSP<individual_size>> p(population_size);

    for(int ii = 0; ii < 5; ii++){
        int x = -1;
        p.set_current_best(9999999);
        p.random(0, 0);
        p.random(-5.12f, 5.12f, population_size / 2);
        p.fitness();
        
        int number_of_epochs = 50000;
        for(int i = 0; i < number_of_epochs; i++){          
            if(i % 5000 == 0){
                int best = p.get_best_value();
                outFile << best << " ";
                if(x == -1 && best <= good_solution){
                    x = i;
                }
            }
            p.mutate(MUTATION_INVERSION, mutation_probability); // 3% for mutation
            p.crossover(CROSSOVER_OWN, crossover_probability); // 30% of the best organisms
            p.sortAll(); // sort all organisms and make selection
        }
        int best = p.get_best_value();
        if(x == -1 && best <= good_solution){
            x = number_of_epochs;
        }
        outFile << best << " | " << x << "\n";
        p.reset_children();
    }
    outFile.close();
    CUDA_CALL(cudaFree(ah), "cudaFree ah");

    return 0;
}