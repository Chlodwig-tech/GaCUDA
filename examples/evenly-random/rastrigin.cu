#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <nvml.h>
#include <cstring>
#include "../../gacuda/gacuda.h"

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


int main(int argc, char *argv[]){
    float good_solution = 0.999f;
    
    const int individual_size = 2; // individual size
    const int population_size = std::atoi(argv[1]); // population size
    float mutation_probability = std::atoi(argv[2]);
    float crossover_probability = std::atoi(argv[3]);
    const char* target_uuid = argv[4];
    nvmlReturn_t result;
    nvmlDevice_t device;
    int num_devices;
    int cuda_device_id = -1;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    cudaError_t cuda_result = cudaGetDeviceCount(&num_devices);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(cuda_result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    for (int i = 0; i < num_devices; ++i) {
        char uuid[80];
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(result) << std::endl;
            continue;
        }

        result = nvmlDeviceGetUUID(device, uuid, sizeof(uuid));
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get UUID for device " << i << ": " << nvmlErrorString(result) << std::endl;
            continue;
        }

        if (strcmp(uuid, target_uuid) == 0) {
            cuda_device_id = i;
            break;
        }
    }
    if (cuda_device_id == -1) {
        std::cerr << "No device with UUID " << target_uuid << " was found." << std::endl;
        nvmlShutdown();
        return 1;
    }
    cuda_result = cudaSetDevice(cuda_device_id);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(cuda_result) << std::endl;
        nvmlShutdown();
        return 1;
    }
    nvmlShutdown();
    std::stringstream filename;
    filename << "results/rastrigin/output-" << population_size << "-" << mutation_probability << "-" << crossover_probability << ".txt";
    std::ofstream outFile(filename.str());
    outFile << std::fixed << std::setprecision(6);

    Population<Rastrigin<individual_size>> p(population_size);

    for(int ii = 0; ii < 5; ii++){
        int x = -1;
        //p.random(-5.12f, 5.12f);
        p.plinspace(-5.12f, 5.12f);
        p.random(-5.12f, 5.12f, population_size / 2);
        p.set_current_best(9999.0f);
        p.fitness();

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
            p.crossover(CROSSOVER_UNIFORM, crossover_probability);
            p.sortAll();
        }
        float best = p.get_best_value();
        if(x == -1 && best <= good_solution){
            x = number_of_epochs;
        }
        outFile << best << " | " << x << "\n";
        p.reset_children();
    }
    outFile.close();
    cudaDeviceSynchronize();
}