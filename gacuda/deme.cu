#ifndef DEME_CU
#define DEME_CU

#include <random>

#include "population.cu"

template<typename T, int Deme_num> class Deme : public Population<T>{
public:
    Deme(int size) : Population<T>(size){}

    void deme_migrate(){}

    void deme_sort();
    void deme_mutate(MUTATION mutation_types[Deme_num], float probability=1.0f);
    void deme_crossover(CROSSOVER crossover_types[Deme_num], float probability=1.0f);
    void deme_printP(int max=-1);
};

template<typename T, int Deme_num> class IslandDeme : public Deme<T, Deme_num>{
public:
    IslandDeme(int size) : Deme<T, Deme_num>(size){}
    void deme_migrate();
};

template<typename T, int Deme_num> class RingDeme : public Deme<T, Deme_num>{
public:
    RingDeme(int size) : Deme<T, Deme_num>(size){}
    void deme_migrate();
};



template<typename T, int Deme_num> void Deme<T, Deme_num>::deme_sort(){
    int deme_size = this->size / Deme_num;
    for(int i = 0; i < Deme_num; i++){
        for(int k = 2; k <= 2 * deme_size; k <<= 1){
            for(int j = k >> 1; j > 0; j >>= 1){
                SortAllKernel<<<deme_size / 1024 + 1, 1024, 0, this->stream>>>(this->porganisms + i * deme_size, this->pchildren + i * deme_size, this->ichildren + i * deme_size, deme_size, j, k);
            }
        }
    }
}

template<typename T, int Deme_num> void Deme<T, Deme_num>::deme_mutate(MUTATION mutation_types[Deme_num], float probability){
    // other streams???
    int deme_size = this->size / Deme_num;
    for(int i = 0; i < Deme_num; i++){
        switch (mutation_types[i])
        {
            case MUTATION_INVERSION:
                MutationInversionKernel<<<deme_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, deme_size, probability, time(NULL));
                break;
            case MUTATION_OWN:
                MutationOwnKernel<<<deme_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, deme_size, probability, time(NULL));
                break;
            case MUTATION_SCRAMBLE:
                MutationScrambleKernel<<<deme_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, deme_size, probability, time(NULL));        
                break;
            case MUTATION_SWAP:
                MutationSwapKernel<<<deme_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, deme_size, probability, time(NULL));
                break;
        }
    }
}

template<typename T, int Deme_num> void Deme<T, Deme_num>::deme_crossover(CROSSOVER crossover_types[Deme_num], float probability){
    int deme_size = this->size / Deme_num;
    int children_size = deme_size * probability / 100;
    for(int i = 0; i < Deme_num; i++){
        switch (crossover_types[i])
        {
            case CROSSOVER_ARITHMETIC:
                CrossoverArithmeticKernel<<<children_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, this->children + i * deme_size, this->ichildren + i * deme_size, children_size, time(NULL));
                break;
            case CROSSOVER_OWN:
                CrossoverOwnKernel<<<children_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, this->children + i * deme_size, this->ichildren + i * deme_size, children_size, time(NULL));
                break;
            case CROSSOVER_SINGLE_POINT:
                CrossoverSinglePointKernel<<<children_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, this->children + i * deme_size, this->ichildren + i * deme_size, children_size, time(NULL));
                break;
            case CROSSOVER_TWO_POINT:
                CrossoverTwoPointKernel<<<children_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, this->children + i * deme_size, this->ichildren + i * deme_size, children_size, time(NULL));
                break;
            case CROSSOVER_UNIFORM:
                CrossoverUniformKernel<<<children_size / 1024 + 1, 1024, 0, this->stream>>>(this->organisms + i * deme_size, this->children + i * deme_size, this->ichildren + i * deme_size, children_size, time(NULL));
                break;
        }
    }
}

template<typename T, int Deme_num> void Deme<T, Deme_num>::deme_printP(int max){
    cudaDeviceSynchronize();
    int deme_size = this->size / Deme_num;
    for(int i = 0; i < Deme_num; i++){            
        printf("Demo %d\n", i);
        PrintPointersKernel<<<1, 1, 0, this->stream>>>(this->porganisms + i * deme_size, deme_size, max == -1 ? deme_size:max);
        cudaDeviceSynchronize();
    }
}

template<typename T, int Deme_num> void IslandDeme<T, Deme_num>::deme_migrate(){
    int deme_size = this->size / Deme_num;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, Deme_num - 1);

    int migrations[Deme_num];
    int *dmigrations;

    for(int i = 0; i < Deme_num; i++){
        migrations[i] = -1;
    }
    for(int i = 0; i < Deme_num; i++){
        int random_deme = distrib(gen);
        if(migrations[random_deme] == -1){
            migrations[i] = random_deme;
            migrations[random_deme] = i;
        }
    }

    CUDA_CALL(cudaMalloc((void **)&dmigrations, Deme_num * sizeof(int)), "IslandDeme migration dmigrations cudaMalloc");
    CUDA_CALL(cudaMemcpy(dmigrations, migrations, Deme_num * sizeof(int), cudaMemcpyHostToDevice), "IslandDeme migration dmigrations cudaMemcpy");
    DemeMigrateKernel<<<Deme_num, 1024, 0, this->stream>>>(this->porganisms, dmigrations, Deme_num, deme_size);
    CUDA_CALL(cudaFree(dmigrations), "IslandDeme migration dmigrations cudaFree");
}

template<typename T, int Deme_num> void RingDeme<T, Deme_num>::deme_migrate(){
    int deme_size = this->size / Deme_num;

    int migrations[Deme_num];
    int *dmigrations;

    migrations[0] = Deme_num - 1;
    for(int i = 1; i < Deme_num - 1; i++){
        migrations[i] = i + 1;
    }

    CUDA_CALL(cudaMalloc((void **)&dmigrations, Deme_num * sizeof(int)), "RingDeme migration dmigrations cudaMalloc");
    CUDA_CALL(cudaMemcpy(dmigrations, migrations, Deme_num * sizeof(int), cudaMemcpyHostToDevice), "RingDeme migration dmigrations cudaMemcpy");
    DemeMigrateKernel<<<Deme_num, 1024, 0, this->stream>>>(this->porganisms, dmigrations, Deme_num, deme_size);
    CUDA_CALL(cudaFree(dmigrations), "RingDeme migration dmigrations cudaFree");
}

#endif // DEME_CU
