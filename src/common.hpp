// this file contains global variables and common functions
#pragma once

#include <cuda.h>

unsigned int batch_size, seq_len, feat_dim_in, feat_dim_out, w_feat_dim_in;

__device__ int binary_search(float* cmf, int cnt, float selected_cmf) {
    // binary search time!
    // we want to find the first value that compares greater than selected_cmf

    int lower = 0, upper = cnt;
    while(upper - lower > 1) {
        // minus one, we always want to try the smaller value first
        int mid = (lower + upper - 1) / 2;
        if(cmf[mid] > selected_cmf) {
            // this *could* be the selected cmf value, set upper to point to it (add one because it is non exclusive)
            upper = mid + 1;
        } else {
            // this is less than or equal to selected cmf
            // it cannot be lower, add one to lower
            lower = mid + 1; 
        }
    }

    return lower;
}


__device__ unsigned int taus_step(unsigned int& z, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int m) {
    unsigned int b = (((z << s1) ^ z) >> s2);
    z = (((z & m) << s3) ^ b);
    return z;
}

__device__ unsigned int lcg_step(unsigned int& z, unsigned int a, unsigned int c) {
    z = a * z + c;
    return z;
}

__device__ unsigned int hybrid_taus_integer(uint4& state) {
    return
        taus_step(state.x, 13, 19, 12, 4294967294U) ^
        taus_step(state.y, 2, 25, 4, 4294967288U) ^
        taus_step(state.z, 3, 11, 17, 4294967280U) ^
        lcg_step(state.w, 1664525, 1013904223U);
}

__device__ float hybrid_taus(uint4& state) {
    return 2.3283064365387e-10 * float(
        hybrid_taus_integer(state)
    );
}
