#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#include "../../gacuda/gacuda.h"

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


int main(int argc, char *argv[]){
    float good_solution = 0.999f;
   

    const int individual_size = 8; // individual size
    const int population_size = std::atoi(argv[1]); // population size
    float mutation_probability = std::atoi(argv[2]);
    float crossover_probability = std::atoi(argv[3]);

    std::stringstream filename;
    filename << "results/root_finder/output-" << population_size << "-" << mutation_probability << "-" << crossover_probability << ".txt";
    std::ofstream outFile(filename.str());
    outFile << std::fixed << std::setprecision(6);

    
    Population<RootFinder<individual_size>> p(population_size);

    for(int ii = 0; ii < 5; ii++){
        int x = -1;
        p.plinspace(-10.0f, 10.0f);
        p.random(-10.0f, 10.0f, population_size / 2);
        // p.random(-10.0f, 10.0f);
        p.set_current_best(9999.0f);
        p.fitness();
        //p.printP(2);
        int number_of_epochs = 50000;
        for(int i = 0; i < number_of_epochs; i++){
            if(i % 5000 == 0){
                float best = p.get_best_value();
                outFile << best << " ";
                if(x == -1 && best <= good_solution){
                    x = i;
                }
            }
            p.shift_mutate(0.001f, mutation_probability);
            //p.mutate(MUTATION_SCRAMBLE, mutation_probability);
            p.crossover(CROSSOVER_UNIFORM, crossover_probability); // 20% of the best organisms
            p.sortAll(); // sort all organisms and make selection
        }
        float best = p.get_best_value();
        if(x == -1 && best <= good_solution){
            x = number_of_epochs;
        }
        outFile << best << " | " << x << "\n";
        p.reset_children();
    }
    outFile.close();

    //p.printP(2);
    //p.print_current_best();
    cudaDeviceSynchronize();
}