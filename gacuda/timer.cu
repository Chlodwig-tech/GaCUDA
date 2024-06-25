#ifndef TIMER_CU
#define TIMER_CU

#include "stdio.h"

class Timer{
    cudaEvent_t estart, estop;
    cudaStream_t *stream;
    float milliseconds;
public:
    Timer(cudaStream_t *cstream=NULL){
        cudaEventCreate(&estart);
        cudaEventCreate(&estop);
        milliseconds = 0;
        stream = cstream;
    }
    void start(){
        if(stream != NULL)
            cudaEventRecord(estart, *stream);
        else
            cudaEventRecord(estart);
    }
    void stop(){
        if(stream != NULL)
            cudaEventRecord(estop, *stream);
        else
            cudaEventRecord(estop);
        cudaEventSynchronize(estop);
        cudaEventElapsedTime(&milliseconds, estart, estop);
    }
    float get(){
        return milliseconds;
    }
    void print(){
        printf("Time taken: %fms\n", milliseconds);
    }
};

#endif // TIMER_CU
