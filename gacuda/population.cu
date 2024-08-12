#ifndef SPOPULATION_CU
#define SPOPULATION_CU

#include "kernels.cu"

#define CUDA_CALL(x, message) {if((x) != cudaSuccess) { \
    printf("Error - %s(%d)[%s]: %s\n", __FILE__, __LINE__, message, cudaGetErrorString(x)); \
    exit(EXIT_FAILURE); }}

enum MUTATION{ // mutations
    MUTATION_INVERSION,
    MUTATION_OWN, 
    MUTATION_SCRAMBLE,
    MUTATION_SWAP
};

enum CROSSOVER{ // crossovers
    CROSSOVER_ARITHMETIC,
    CROSSOVER_OWN,
    CROSSOVER_SINGLE_POINT,
    CROSSOVER_TWO_POINT,
    CROSSOVER_UNIFORM,
};

template<typename T> class Population{
protected:
    using DNA = typename T::DNA_t;
    using Tfitness = typename T::Tfitness_t;

    T *organisms;
    T *children;
    T* *porganisms;
    T* *pchildren;
    T helper;
    bool *ichildren;
    int size;
    cudaStream_t stream;

public:
    Population(int size);
    ~Population();
    template<typename r> void random(r a, r b);
    template<typename r> void brandom(r a, r b, int nthreads=1024);
    template<typename r> void linspace(r a, r b, bool endpoint=true);
    template<typename r> void blinspace(r a, r b, bool endpoint=true);
    template<typename r> void logspace(r a, r b, DNA base, bool endpoint=true);
    template<typename r> void blogspace(r a, r b, DNA base, bool endpoint=true);
    template<typename r> void plinspace(r a, r b, bool endpoint=true);
    template<typename r> void bplinspace(r a, r b, bool endpoint=true);
    template<typename r> void plogspace(r a, r b, DNA base, bool endpoint=true);
    template<typename r> void bplogspace(r a, r b, DNA base, bool endpoint=true);
    void fitness();
    void bfitness(int nthreads=1024);
    void mutate(MUTATION mutation_type, float probability=1.0f);
    void crossover(CROSSOVER crossover_type, float probability=1.0f);
    void sortAll();
    void sortOrganisms();
    void init_organisms();
    void init_organisms_with_tid();
    void init_organisms_with_val(DNA val);
    void binit_organisms_with_val(DNA val);

    void printP(int max=-1);
    void print(int max=-1);
    void print_childrenP(int max=-1);
    void print_children(int max=-1);

    cudaStream_t* getStream();
};

template<typename T> Population<T>::Population(int size) : size(size){
    CUDA_CALL(cudaMalloc((void **)&organisms, size * sizeof(T)), "Population organisms cudaMalloc");
    CUDA_CALL(cudaMalloc((void **)&porganisms, size * sizeof(T*)), "Population porganisms cudaMalloc");
    CUDA_CALL(cudaMalloc((void **)&children, size * sizeof(T)), "Population children cudaMalloc");
    CUDA_CALL(cudaMalloc((void **)&pchildren, size * sizeof(T*)), "Population pchildren cudaMalloc");
    CUDA_CALL(cudaMalloc((void **)&ichildren, size * sizeof(bool)), "Population ichildren cudaMalloc");
    InitKernel<<<size / 1024 + 1, 1024>>>(organisms, porganisms, children, pchildren, size);
    CUDA_CALL(cudaStreamCreate(&stream), "Population stream create");
}

template<typename T> Population<T>::~Population(){
    CUDA_CALL(cudaFree(organisms), "Population organisms cudaFree");
    CUDA_CALL(cudaFree(porganisms), "Population porganisms cudaFree");
    CUDA_CALL(cudaFree(children), "Population children cudaFree");
    CUDA_CALL(cudaFree(pchildren), "Population pchildren cudaFree");
    CUDA_CALL(cudaFree(ichildren), "Population ichildren cudaFree");
    CUDA_CALL(cudaStreamDestroy(stream), "Population stream destroy");
}

template<typename T> template<typename r> void Population<T>::random(r a, r b){
    RandomKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, time(NULL), size, a, b);
}

template<typename T> template<typename r> void Population<T>::brandom(r a, r b, int nthreads){
    BRandomKernel<<<size, nthreads, 0, stream>>>(organisms, time(NULL), a, b);
}

template<typename T> template<typename r> void Population<T>::linspace(r a, r b, bool endpoint){
    LinspaceKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, a, b, endpoint);
}

template<typename T> template<typename r> void Population<T>::blinspace(r a, r b, bool endpoint){
    BLinspaceKernel<<<size, 1024, 0, stream>>>(organisms, size, a, b, endpoint);
}

template<typename T> template<typename r> void Population<T>::plinspace(r a, r b, bool endpoint){
    PLinspaceKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, a, b, endpoint);
}

template<typename T> template<typename r> void Population<T>::bplinspace(r a, r b, bool endpoint){
    BPLinspaceKernel<<<size, 1024, 0, stream>>>(organisms, size, a, b, endpoint);
}

template<typename T> template<typename r> void Population<T>::logspace(r a, r b, DNA base, bool endpoint){
    LogspaceKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, a, b, base, endpoint);
}

template<typename T> template<typename r> void Population<T>::blogspace(r a, r b, DNA base, bool endpoint){
    BLogspaceKernel<<<size, 1024, 0, stream>>>(organisms, size, a, b, base, endpoint);
}

template<typename T> template<typename r> void Population<T>::plogspace(r a, r b, DNA base, bool endpoint){
    PLogspaceKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, a, b, base, endpoint);
}

template<typename T> template<typename r> void Population<T>::bplogspace(r a, r b, DNA base, bool endpoint){
    BPLogspaceKernel<<<size, 1024, 0, stream>>>(organisms, size, a, b, base, endpoint);
}

template<typename T> void Population<T>::fitness(){
    FitnessKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size);
}

template<typename T> void Population<T>::bfitness(int nthreads){
    BFitnessKernel<T, Tfitness><<<size, nthreads, nthreads * sizeof(Tfitness), stream>>>(organisms, nthreads);
}

template<typename T> void Population<T>::mutate(MUTATION mutation_type, float probability){
    switch (mutation_type)
    {
        case MUTATION_INVERSION:
            MutationInversionKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, probability, time(NULL));
            break;
        case MUTATION_OWN:
            MutationOwnKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, probability, time(NULL));
            break;
        case MUTATION_SCRAMBLE:
            MutationScrambleKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, probability, time(NULL));        
            break;
        case MUTATION_SWAP:
            MutationSwapKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, probability, time(NULL));
            break;
    }
}

template<typename T> void Population<T>::crossover(CROSSOVER crossover_type, float probability){
    int children_size = size * probability / 100;
    switch (crossover_type)
    {
        case CROSSOVER_ARITHMETIC:
            CrossoverArithmeticKernel<<<children_size / 1024 + 1, 1024, 0, stream>>>(organisms, children, ichildren, children_size, time(NULL));
            break;
        case CROSSOVER_OWN:
            CrossoverOwnKernel<<<children_size / 1024 + 1, 1024, 0, stream>>>(organisms, children, ichildren, children_size, time(NULL));
            break;
        case CROSSOVER_SINGLE_POINT:
            CrossoverSinglePointKernel<<<children_size / 1024 + 1, 1024, 0, stream>>>(organisms, children, ichildren, children_size, time(NULL));
            break;
        case CROSSOVER_TWO_POINT:
            CrossoverTwoPointKernel<<<children_size / 1024 + 1, 1024, 0, stream>>>(organisms, children, ichildren, children_size, time(NULL));
            break;
        case CROSSOVER_UNIFORM:
            CrossoverUniformKernel<<<children_size / 1024 + 1, 1024, 0, stream>>>(organisms, children, ichildren, children_size, time(NULL));
            break;
    }
}

template<typename T> void Population<T>::sortAll(){
    for(int k = 2; k <= 2 * size; k <<= 1){
        for(int j = k >> 1; j > 0; j >>= 1){
            SortAllKernel<<<size / 1024 + 1, 1024, 0, stream>>>(porganisms, pchildren, ichildren, size, j, k);
        }
    }
}

template<typename T> void Population<T>::sortOrganisms(){
    for(int k = 2; k <= 2 * size; k <<= 1){
        for(int j = k >> 1; j > 0; j >>= 1){
            BitonicSortKernel<<<size / 1024 + 1, 1024, 0, stream>>>(porganisms, size, j, k);
        }
    }
}

template<typename T> void Population<T>::init_organisms(){
    InitOrganismsKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size);
}

template<typename T> void Population<T>::init_organisms_with_tid(){
    InitOrganismsWithTidKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size);
}

template<typename T> void Population<T>::init_organisms_with_val(DNA val){
    InitOrganismsWithValKernel<<<size / 1024 + 1, 1024, 0, stream>>>(organisms, size, val);
}

template<typename T> void Population<T>::binit_organisms_with_val(DNA val){
    BInitOrganismsWithValKernel<<<size, 1024, 0, stream>>>(organisms, size, val);
}

template<typename T> void Population<T>::printP(int max){
    PrintPointersKernel<<<1, 1, 0, stream>>>(porganisms, size, max == -1 ? size:max);
}

template<typename T> void Population<T>::print(int max){
    PrintKernel<<<1, 1, 0, stream>>>(organisms, size, max == -1 ? size:max);
}

template<typename T> void Population<T>::print_childrenP(int max){
    PrintChildrenKernelP<<<1, 1, 0, stream>>>(pchildren, ichildren, size, max == -1 ? size:max);
}

template<typename T> void Population<T>::print_children(int max){
    PrintChildrenKernel<<<1, 1, 0, stream>>>(children, ichildren, size, max == -1 ? size:max);
}

template<typename T> cudaStream_t* Population<T>::getStream(){
    return &stream;
}

#endif // SPOPULATION_CU
