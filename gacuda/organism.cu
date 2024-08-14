#ifndef ORGANISM_CU
#define ORGANISM_CU

#include <cuda_runtime.h>
#include <curand_kernel.h>

template<typename DNA, typename Tfitness, int Size>
class Organism{
protected:    
    DNA genes[Size];
    Tfitness fvalue;

public:

    using DNA_t = DNA;
    using Tfitness_t = Tfitness;

    __device__ void fitness(){}
    __device__ Tfitness fitness_part(int tid){}
    __device__ void print(){}
    __device__ void own_mutate(curandState *state){}
    __device__ void own_crossover(curandState *state, Organism *second_parent, Organism *child){}
    __device__ void init(){}
    __device__ void init_with_tid(int tid){}

    __device__ bool is_greater(Organism *other);
    template<typename r> __device__ void random(curandState *state, r a, r b);
    template<typename r> __device__ void brandom(curandState *state, int tid, r a, r b);
    template<typename r> __device__ void linspace(r a, r b, bool endpoint);
    template<typename r> __device__ void blinspace(r a, r b, bool endpoint, int tid);
    template<typename r> __device__ void bplinspace(int size, r a, r b, bool endpoint, int tid, int bid);
    template<typename r> __device__ void plinspace(int size, r a, r b, bool endpoint, int tid);
    template<typename r> __device__ void logspace(r a, r b, DNA base, bool endpoint);
    template<typename r> __device__ void blogspace(r a, r b, DNA base, bool endpoint, int tid);
    template<typename r> __device__ void plogspace(int size, r a, r b, DNA base, bool endpoint, int tid);
    template<typename r> __device__ void bplogspace(int size, r a, r b, DNA base, bool endpoint, int tid, int bid);
    __device__ void init_with_val(DNA val);
    __device__ void binit_with_val(DNA val, int tid);
    __device__ void swap_mutate(curandState *state);
    __device__ void inversion_mutate(curandState *state);
    __device__ void scramble_mutate(curandState *state);
    __device__ void crossover_arithmetic(curandState *state, Organism *second_parent, Organism *child);
    __device__ void crossover_single_point(curandState *state, Organism *second_parent, Organism *child);
    __device__ void crossover_two_point(curandState *state, Organism *second_parent, Organism *child);
    __device__ void crossover_uniform(curandState *state, Organism *second_parent, Organism *child);
    __host__ __device__ int get_size();
    __device__ Tfitness get_fvalue();
    __device__ void Setf(Tfitness fval);
    __device__ void SetfMax();
    __device__ void Set(Organism *other);
};

template<typename DNA, typename Tfitness, int Size> __device__ bool Organism<DNA, Tfitness, Size>::
is_greater(Organism *other){
    return this->fvalue > other->fvalue;
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
random(curandState *state, r a, r b){
    for(int i = 0; i < Size; i++){
        this->genes[i] = curand_uniform(state) * (b - a) + a;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
brandom(curandState *state, int tid, r a, r b){
    for(int i = tid; i < Size; i += 1024){
        this->genes[i] = curand(state) % (b - a) + a;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
linspace(r a, r b, bool endpoint){
    r s = (b - a) / (Size - (endpoint ? 1:0));
    DNA x = a;
    for(int i = 0; i < Size; i++){
        this->genes[i] = x;
        x += s;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
blinspace(r a, r b, bool endpoint, int tid){
    r s = (b - a) / (Size - (endpoint ? 1:0));
    for(int i = tid; i < Size; i+=1024){
        this->genes[i] = a + s * i;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
bplinspace(int size, r a, r b, bool endpoint, int tid, int bid){
    r s = (b - a) / (Size * size - (endpoint ? 1:0));
    for(int i = tid; i < Size; i+=1024){
        this->genes[i] = a + s * bid * Size + s * i;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
plinspace(int size, r a, r b, bool endpoint, int tid){
    r s = (b - a) / (Size * size - (endpoint ? 1:0));
    DNA x = a + s * tid * Size;
    for(int i = 0; i < Size; i++){
        this->genes[i] = x;
        x += s;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
logspace(r a, r b, DNA base, bool endpoint){
    r s = (b - a) / (Size - (endpoint ? 1:0));
    DNA x = a;
    for(int i = 0; i < Size; i++){
        this->genes[i] = pow(base, x);
        x += s;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
blogspace(r a, r b, DNA base, bool endpoint, int tid){
    r s = (b - a) / (Size - (endpoint ? 1:0));
    for(int i = tid; i < Size; i+=1024){
        this->genes[i] = pow(base, s * i);
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
plogspace(int size, r a, r b, DNA base, bool endpoint, int tid){
    r s = (b - a) / (Size * size - (endpoint ? 1:0));
    DNA x = a + s * tid * Size;
    for(int i = 0; i < Size; i++){
        this->genes[i] = pow(base, x);
        x += s;
    }
}

template<typename DNA, typename Tfitness, int Size> template<typename r> __device__ void Organism<DNA, Tfitness, Size>::
bplogspace(int size, r a, r b, DNA base, bool endpoint, int tid, int bid){
    r s = (b - a) / (Size * size - (endpoint ? 1:0));
    for(int i = tid; i < Size; i+=1024){
        this->genes[i] = pow(base, a + s * bid * Size + s * i);
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
init_with_val(DNA val){
    for(int i = 0; i < Size; i++){
        this->genes[i] = val;
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
binit_with_val(DNA val, int tid){
    for(int i = tid; i < Size; i+=1024){
        this->genes[i] = val;
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
swap_mutate(curandState *state){
    int i = curand(state) % Size;
    int j = curand(state) % Size;

    DNA temp = this->genes[i];
    this->genes[i] = this->genes[j];
    this->genes[j] = temp;
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
inversion_mutate(curandState *state){
    int i = curand(state) % Size;
    int j = curand(state) % Size;

    int start = i < j ? i:j;
    int end = i < j ? j:i;

    while(start <= end){
        DNA temp = this->genes[start];
        this->genes[start] = this->genes[end];
        this->genes[end] = temp;
        start++;
        end--;
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
scramble_mutate(curandState *state){
    int i = curand(state) % Size;
    int j = curand(state) % Size;

    int start = i < j ? i:j;
    int end = i < j ? j:i;
    
    for(int i = start; i <= end; i++){
        int j = curand_uniform(state) * (end - start) + start;
        DNA temp = this->genes[i];
        this->genes[i] = this->genes[j];
        this->genes[j] = temp;
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
crossover_arithmetic(curandState *state, Organism *second_parent, Organism *child){
    for(int i = 0; i < Size; i++){
        DNA temp = this->genes[i] + second_parent->genes[i];
        child->genes[i] = (DNA)(temp / 2);
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
crossover_single_point(curandState *state, Organism *second_parent, Organism *child){
    int part = curand(state) % Size;
    for(int i = 0; i < part; i++){
        child->genes[i] = this->genes[i];
    }
    for(int i = part; i < Size; i++){
        child->genes[i] = second_parent->genes[i];
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
crossover_two_point(curandState *state, Organism *second_parent, Organism *child){
    int a = curand(state) % Size;
    int b = curand(state) % Size;

    int part1 = a < b ? a:b;
    int part2 = a < b ? b:a;
    
    for(int i = 0; i < part1; i++){
        child->genes[i] = this->genes[i];
    }
    for(int i = part1; i < part2; i++){
        child->genes[i] = second_parent->genes[i];
    }
    for(int i = part2; i < Size; i++){
        child->genes[i] = this->genes[i];
    }
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
crossover_uniform(curandState *state, Organism *second_parent, Organism *child){
    for(int i = 0; i < Size; i++){
        int a = curand(state) % 2;
        child->genes[i] = a == 0 ? this->genes[i] : second_parent->genes[i];
    }
}

template<typename DNA, typename Tfitness, int Size> __host__ __device__ int Organism<DNA, Tfitness, Size>::
get_size(){
    return Size;
}

template<typename DNA, typename Tfitness, int Size> __device__ Tfitness Organism<DNA, Tfitness, Size>::
get_fvalue(){
    return fvalue;
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
Setf(Tfitness fval){
    this->fvalue = fval;
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
SetfMax(){
    this->fvalue = (Tfitness)(~0 & ~(1 << (sizeof(Tfitness) * 8 - 1)));
}

template<typename DNA, typename Tfitness, int Size> __device__ void Organism<DNA, Tfitness, Size>::
Set(Organism *other){
    for(int i = 0; i < Size; i++){
        this->genes[i] = other->genes[i];
    }
    this->fvalue = other->fvalue;
}

#endif // ORGANISM_CU
