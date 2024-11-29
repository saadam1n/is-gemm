#pragma once

#include <cuda.h>
#include <torch/script.h>
#include <iostream>

#include "common.hpp"

__global__ void ais_gemm_kernel(
    unsigned int num_rows, 
    unsigned int feat_dim_in, 
    unsigned int feat_dim_out,
    unsigned int num_accumulations,
    unsigned int seed,
    float* location_cmf, 
    float* input,
    float* input_cmf,
    float* weight,
    float* weight_cmf,
    float* output
) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // set up the a PRNG state using prime numbers
    // primes copied from https://www.math.utah.edu/~pa/MDS/primes.html
    unsigned int thread_seed = index + seed;
    uint4 state = {
        thread_seed * 2147483647 + 2147483629, 
        thread_seed * 2147483563 + 2147483549, 
        thread_seed * 2147483489 + 2147483477, 
        thread_seed * 2147483353 + 2147483323
    };

    for(int i = 0; i < num_accumulations; i++) {
        // generate a random number in [0, 1) and binary search its associated location
        float selected_location_cmf = hybrid_taus(state);
        int location = binary_search(location_cmf, num_rows * feat_dim_out, selected_location_cmf);

        int row_idx = location / feat_dim_out;
        int col_idx = location % feat_dim_out;

        float dot_prod_cmf = 2 * hybrid_taus(state);

        // binary search both lists at time
        int lower = 0, upper = feat_dim_in;
        while(upper - lower > 1) {
            int mid = (lower + upper - 1) / 2;

            float combined_cmf = input_cmf[row_idx * feat_dim_in + mid] + weight_cmf[col_idx * feat_dim_in + mid];

            if(combined_cmf > dot_prod_cmf) {
                upper = mid + 1;
            } else {
                lower = mid + 1; 
            }
        }


        float double_dot_prod_pmf = input_cmf[row_idx * feat_dim_in + lower] + weight_cmf[col_idx * feat_dim_in + lower] 
                            - (lower != 0 ? input_cmf[row_idx * feat_dim_in + lower - 1] + weight_cmf[col_idx * feat_dim_in + lower - 1] : 0.0);

        float term = input[row_idx * feat_dim_in + lower] * weight[col_idx * feat_dim_in + lower] / double_dot_prod_pmf;
        atomicAdd(&output[row_idx * feat_dim_out + col_idx], term);
    }
}

#define SVD_PRIOR_ESTIMATION

torch::Tensor run_ais_gemm(torch::Tensor input, torch::Tensor weight) {
    // for debugging pruposes I will use the exact reference matrix as the prior estimate
    #ifdef SVD_PRIOR_ESTIMATION
    auto svd = torch::linalg_svd(weight.transpose(0, 1), false);
    torch::Tensor u  = std::get<0>(svd);
    torch::Tensor s  = torch::diagflat(std::get<1>(svd));
    torch::Tensor vh = std::get<2>(svd);

    torch::Tensor lhs = torch::matmul(input, u);
    torch::Tensor rhs = torch::matmul(s, vh);
    torch::Tensor prior_estimate = torch::matmul(lhs, rhs);
    #else
    torch::Tensor prior_estimate = torch::matmul(input, weight.transpose(0, 1));
    #endif
    prior_estimate = prior_estimate.view(batch_size * seq_len * feat_dim_out).contiguous().abs();
    torch::Tensor location_cmf = prior_estimate.cumsum(0);
    float location_norm_factor = location_cmf[batch_size * seq_len * feat_dim_out - 1].item<float>();
    location_cmf = location_cmf / location_norm_factor;
    torch::Tensor location_pmf = prior_estimate / location_norm_factor;

    // create the inner product CMFs
    torch::Tensor input_cmf = input.abs().cumsum(1);
    input_cmf = input_cmf / input_cmf.narrow(1, feat_dim_in - 1, 1).expand({-1, feat_dim_in});
    input_cmf = input_cmf.contiguous(); 

    torch::Tensor weight_cmf = weight.abs().cumsum(1);
    weight_cmf = weight_cmf / weight_cmf.narrow(1, feat_dim_in - 1, 1).expand({-1, feat_dim_in});
    weight_cmf = weight_cmf.contiguous(); 


    // now that are tensors are set up, let's do the actual accumulation
    const int num_accumulations = 1024;
    // these parameters fill up a RTX A5000 GPU
    const int block_size = 768; // 2 blocks per SM
    const int num_threads = block_size * 128; // 64 SMs per GPU, 2 * 64 = 128

    std::cout << "Running assisted importance sampling pass...\n";

    torch::Tensor output = torch::zeros({batch_size * seq_len, feat_dim_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();

    cudaDeviceSynchronize();
    ais_gemm_kernel<<<num_threads / block_size, block_size>>>(
        batch_size * seq_len,
        feat_dim_in,
        feat_dim_out,
        num_accumulations,
        time(nullptr),
        (float*)location_cmf.data_ptr(),
        (float*)input.data_ptr(), 
        (float*)input_cmf.data_ptr(), 
        (float*)weight.data_ptr(), 
        (float*)weight_cmf.data_ptr(), 
        (float*)output.data_ptr()
    );
    cudaDeviceSynchronize();

    float norm_factor = (2.0 / double(num_threads)) / double(num_accumulations) / 8;
    output = norm_factor * output / location_pmf.view({batch_size * seq_len, feat_dim_out});

    // placeholder return value
    return output;
}