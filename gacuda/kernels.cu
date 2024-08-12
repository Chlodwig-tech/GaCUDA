#ifndef KERNELS_CU
#define KERNELS_CU

// __DEVICE__ FUNCTIONS DECLARATIONS

template<typename T>__device__
bool isgreater(T* *porganisms, T* *pchildren, bool *ichildren, int size, int tid, int ixj);


// KERNELS DECLARATIONS

template<typename T, typename Tfitness> __global__ void BFitnessKernel(T *organisms, int nthreads);
template<typename T> __global__ void BitonicSortKernel(T *organisms, int size, int j, int k);
template<typename T, typename r> __global__ void BRandomKernel(T *organisms, unsigned long long seed, r a, r b);
template<typename T, typename DNA> __global__ void BInitOrganismsWithValKernel(T *organisms, int size, DNA val);
template<typename T, typename r> __global__ void BLinspaceKernel(T *organisms, int size, r a, r b, bool endpoint);
template<typename T, typename r, typename DNA> __global__ void BLogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint);
template<typename T> __global__ void CrossoverArithmeticKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed);
template<typename T> __global__ void CrossoverOwnKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed);
template<typename T> __global__ void CrossoverSinglePointKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed);
template<typename T> __global__ void CrossoverTwoPointKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed);
template<typename T> __global__ void CrossoverUniformKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed);
template<typename T> __global__ void DemeIslandMigrateKernel(T* *porganisms, int *migrations, int Deme_num, int deme_size);
template<typename T> __global__ void DemeRingMigrateKernel(T* *porganisms, int Deme_num, int deme_size);
template<typename T> __global__ void DemeSteppingStoneMigrateKernel(T* *porganisms, int Deme_num, int deme_size);
template<typename T> __global__ void FitnessKernel(T *organisms, int size);
template<typename T> __global__ void InitKernel(T *organisms, T* *porganisms, T *children, T* *pchildren, int size);
template<typename T> __global__ void InitOrganismsKernel(T *organisms, int size);
template<typename T> __global__ void InitOrganismsWithTidKernel(T *organisms, int size);
template<typename T, typename DNA> __global__ void InitOrganismsWithValKernel(T *organisms, int size, DNA val);
template<typename T, typename r> __global__ void LinspaceKernel(T *organisms, int size, r a, r b, bool endpoint);
template<typename T, typename r, typename DNA> __global__ void LogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint);
template<typename T> __global__ void MutationInversionKernel(T *organisms, int size, float probability, unsigned long long seed);
template<typename T> __global__ void MutationOwnKernel(T *organisms, int size, float probability, unsigned long long seed);
template<typename T> __global__ void MutationScrambleKernel(T *organisms, int size, float probability, unsigned long long seed);
template<typename T> __global__ void MutationSwapKernel(T *organisms, int size, float probability, unsigned long long seed);
template<typename T, typename r> __global__ void BPLinspaceKernel(T *organisms, int size, r a, r b, bool endpoint);
template<typename T, typename r> __global__ void PLinspaceKernel(T *organisms, int size, r a, r b, bool endpoint);
template<typename T, typename r, typename DNA> __global__ void BPLogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint);
template<typename T, typename r, typename DNA> __global__ void PLogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint);
template<typename T> __global__ void PrintChildrenKernel(T *children, bool *indexes, int size, int max);
template<typename T> __global__ void PrintChildrenKernelP(T* *pchildren, bool *indexes, int size, int max);
template<typename T> __global__ void PrintKernel(T *organisms, int size, int max);
template<typename T> __global__ void PrintPointersKernel(T* *porganisms, int size, int max);
template<typename T, typename r> __global__ void RandomKernel(T *organisms, unsigned long long seed, int size, r a, r b);
template<typename T> __global__ void SortAllKernel(T* *porganisms, T* *pchildren, bool *ichildren, int size, int j, int k);

// __DEVICE__ FUNCTIONS

template<typename T>__device__
bool isgreater(T* *porganisms, T* *pchildren, bool *ichildren, int size, int tid, int ixj){
    if(tid < size){
        if(ixj < size){
            return porganisms[tid]->is_greater(porganisms[ixj]);
        }
        if(ichildren[ixj - size]){
            return porganisms[tid]->is_greater(pchildren[ixj - size]);
        }
        return false;
    }
    if(!ichildren[tid - size]){
        return true;
    }
    if(ixj < size){
        return pchildren[tid - size]->is_greater(porganisms[ixj]);
    }
    if(ichildren[ixj - size]){
        return pchildren[tid - size]->is_greater(pchildren[ixj - size]);
    }
    return false;
}


// KERNELS

template<typename T, typename Tfitness> __global__
void BFitnessKernel(T *organisms, int nthreads){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    extern __shared__ Tfitness shared_sum[];
    shared_sum[tid] = organisms[bid].fitness_part(tid);
    __syncthreads();
    for(int i = 1; i < nthreads; i *= 2){
        if(tid % (2 * i) == 0){
            shared_sum[tid] += shared_sum[tid + i];
        }
        __syncthreads();
    }
    if(tid == 0){
        organisms[bid].Setf(shared_sum[0]);
    }
}

template<typename T> __global__ 
void BitonicSortKernel(T* *porganisms, int size, int j, int k){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = tid ^ j;
    if(tid < size && ixj < size && ixj > tid && 
        !(((tid & k) == 0) ^ (porganisms[tid]->is_greater(porganisms[ixj])))
    )
    {
        T *temp = porganisms[tid];
        porganisms[tid] = porganisms[ixj];
        porganisms[ixj] = temp;
    }
}

template<typename T, typename r> __global__ 
void BRandomKernel(T *organisms, unsigned long long seed, r a, r b){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    curandState state;
    curand_init(seed + bid, tid, 0, &state);
    organisms[bid].brandom(&state, tid, a, b);
}

template<typename T, typename DNA> __global__ 
void BInitOrganismsWithValKernel(T *organisms, int size, DNA val){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    organisms[bid].binit_with_val(val, tid);
}

template<typename T, typename r> __global__ 
void BLinspaceKernel(T *organisms, int size, r a, r b, bool endpoint){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    organisms[bid].blinspace(a, b, endpoint, tid);
}

template<typename T, typename r, typename DNA> __global__ 
void BLogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    organisms[bid].blogspace(a, b, endpoint, tid);
}

template<typename T> __global__ 
void CrossoverArithmeticKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        ichildren[tid] = true;
        int parent_index = curand(&state) % size;
        organisms[tid].crossover_arithmetic(&state, &organisms[parent_index], &children[tid]);
        children[tid].fitness();
    }else{
        ichildren[tid] = false;
    }
}

template<typename T> __global__ 
void CrossoverOwnKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        ichildren[tid] = true;
        int parent_index = curand(&state) % size;
        organisms[tid].own_crossover(&state, &organisms[parent_index], &children[tid]);
        children[tid].fitness();
    }else{
        ichildren[tid] = false;
    }
}

template<typename T> __global__ 
void CrossoverSinglePointKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        ichildren[tid] = true;
        int parent_index = curand(&state) % size;
        organisms[tid].crossover_single_point(&state, &organisms[parent_index], &children[tid]);
        children[tid].fitness();
    }else{
        ichildren[tid] = false;
    }
}

template<typename T> __global__ 
void CrossoverTwoPointKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        ichildren[tid] = true;
        int parent_index = curand(&state) % size;
        organisms[tid].crossover_two_point(&state, &organisms[parent_index], &children[tid]);
        children[tid].fitness();
    }else{
        ichildren[tid] = false;
    }
}

template<typename T> __global__ 
void CrossoverUniformKernel(T *organisms, T *children, bool *ichildren, int size, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        ichildren[tid] = true;
        int parent_index = curand(&state) % size;
        organisms[tid].crossover_uniform(&state, &organisms[parent_index], &children[tid]);
        children[tid].fitness();
    }else{
        ichildren[tid] = false;
    }
}

template<typename T> __global__ 
void DemeIslandMigrateKernel(T* *porganisms, int *migrations, int Deme_num, int deme_size){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    while(tid < deme_size && migrations[bid] != -1 && bid < migrations[bid]){
        T *temp = porganisms[bid * deme_size + tid];
        porganisms[bid * deme_size + tid] = porganisms[migrations[bid] * deme_size + tid];
        porganisms[migrations[bid] * deme_size + tid] = temp;
        tid += 1024;
    }
}

template<typename T> __global__ 
void DemeRingMigrateKernel(T* *porganisms, int Deme_num, int deme_size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < deme_size){
        T *helper = porganisms[(Deme_num - 1) * deme_size + tid];
        for(int i = 0; i < Deme_num; i++){
            T *temp = porganisms[i * deme_size + tid];
            porganisms[i * deme_size + tid] = helper;
            helper = temp;
        }
    }
}

template<typename T> __global__ 
void DemeSteppingStoneMigrateKernel(T* *porganisms, int Deme_num, int deme_size){
    int tid = threadIdx.x;
    int bid = blockIdx.x + 1;
    int half_size = deme_size / 2;
    while(tid < half_size){
        T *temp = porganisms[bid * deme_size + tid];
        porganisms[bid * deme_size + tid] = porganisms[(bid - 1) * deme_size + tid + half_size];
        porganisms[(bid - 1) * deme_size + tid + half_size] = temp;
        tid += 1024;
    }
}

template<typename T> __global__ 
void FitnessKernel(T *organisms, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].fitness();
    }
}

template<typename T> __global__ 
void InitKernel(T *organisms, T* *porganisms, T *children, T* *pchildren, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        porganisms[tid] = &organisms[tid];
        pchildren[tid] = &children[tid];
    }
}

template<typename T> __global__ 
void InitOrganismsKernel(T *organisms, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].init();
    }
}

template<typename T> __global__ 
void InitOrganismsWithTidKernel(T *organisms, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].init_with_tid(tid);
    }
}

template<typename T, typename DNA> __global__ 
void InitOrganismsWithValKernel(T *organisms, int size, DNA val){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].init_with_val(val);
    }
}

template<typename T, typename r> __global__ 
void LinspaceKernel(T *organisms, int size, r a, r b, bool endpoint){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].linspace(a, b, endpoint);
    }
}

template<typename T, typename r, typename DNA> __global__ 
void LogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].logspace(a, b, base, endpoint);
    }
}

template<typename T> __global__ 
void MutationInversionKernel(T *organisms, int size, float probability, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        if(curand_uniform(&state) * 100.0f < probability){
            organisms[tid].inversion_mutate(&state);
            organisms[tid].fitness();
        }
    }
}

template<typename T> __global__ 
void MutationOwnKernel(T *organisms, int size, float probability, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        if(curand_uniform(&state) * 100.0f < probability){
            organisms[tid].own_mutate(&state);
            organisms[tid].fitness();
        }
    }
}

template<typename T> __global__ 
void MutationScrambleKernel(T *organisms, int size, float probability, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        if(curand_uniform(&state) * 100.0f < probability){
            organisms[tid].scramble_mutate(&state);
            organisms[tid].fitness();
        }
    }
}

template<typename T> __global__ 
void MutationSwapKernel(T *organisms, int size, float probability, unsigned long long seed){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        if(curand_uniform(&state) * 100.0f < probability){
            organisms[tid].swap_mutate(&state);
            organisms[tid].fitness();
        }
    }
}

template<typename T, typename r> __global__
void BPLinspaceKernel(T *organisms, int size, r a, r b, bool endpoint){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    organisms[bid].bplinspace(size, a, b, endpoint, tid, bid);
}

template<typename T, typename r> __global__
void PLinspaceKernel(T *organisms, int size, r a, r b, bool endpoint){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].plinspace(size, a, b, endpoint, tid);
    }
}

template<typename T, typename r, typename DNA> __global__ 
void BPLogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    organisms[bid].bplogspace(size, a, b, base, endpoint, tid, bid);
}

template<typename T, typename r, typename DNA> __global__ 
void PLogspaceKernel(T *organisms, int size, r a, r b, DNA base, bool endpoint){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        organisms[tid].plogspace(size, a, b, base, endpoint, tid);
    }
}

template<typename T> __global__ 
void PrintChildrenKernel(T *children, bool *indexes, int size, int max){
    for(int i = 0; i < size && i < max; i++){
        if(indexes[i])
            children[i].print();
    }
    printf("\n");
}

template<typename T> __global__ 
void PrintChildrenKernelP(T* *pchildren, bool *indexes, int size, int max)
{
    for(int i = 0; i < size && i < max; i++){
        if(indexes[i])
            pchildren[i]->print();
    }
    printf("\n");
}

template<typename T> __global__ 
void PrintKernel(T *organisms, int size, int max)
{
    for(int i = 0; i < size && i < max; i++){
        organisms[i].print();
    }
    printf("\n");
}

template<typename T> __global__ 
void PrintPointersKernel(T* *porganisms, int size, int max)
{
    for(int i = 0; i < size && i < max; i++){
        porganisms[i]->print();
    }
    printf("\n");
}

template<typename T, typename r> __global__ 
void RandomKernel(T *organisms, unsigned long long seed, int size, r a, r b){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
        curandState state;
        curand_init(seed, tid, 0, &state);
        organisms[tid].random(&state, a, b);
    }
}

template<typename T> __global__ 
void SortAllKernel(T* *porganisms, T* *pchildren, bool *ichildren, int size, int j, int k){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ixj = tid ^ j;
    if(tid < 2 * size && ixj < 2 * size && ixj > tid &&
        !(((tid & k) == 0) ^ (isgreater(porganisms, pchildren, ichildren, size, tid, ixj)))
    )
    {
        if(tid < size){
            T *temp = porganisms[tid];
            if(ixj < size){
                porganisms[tid] = porganisms[ixj];
                porganisms[ixj] = temp;
            }else{
                porganisms[tid] = pchildren[ixj - size];
                pchildren[ixj - size] = porganisms[tid];
                ichildren[ixj - size] = true;
            }
        }else{
            T *temp = pchildren[tid - size];
            if(ixj < size){
                pchildren[tid - size] = porganisms[ixj];
                porganisms[ixj] = temp;
                ichildren[tid - size] = true;
            }else{
                pchildren[tid - size] = pchildren[ixj - size];
                pchildren[ixj - size] = temp;
                bool btemp = ichildren[tid - size];
                ichildren[tid - size] = ichildren[ixj - size];
                ichildren[ixj - size] = btemp;
            }
        }
        
    }
}

#endif // KERNELS_CU
