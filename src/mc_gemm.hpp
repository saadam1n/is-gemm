#pragma once

#include <cuda.h>
#include <torch/script.h>
#include <iostream>

__global__ void mc_gemm_kernel(
    int num_rows, int feat_dim_in, int feat_dim_out, int num_accumulations, 
    float* input, float* weight,
    float* output
) {
    // prime numbers for our LCG
    const int LCG_MULT = 999983;
    const int LCG_ITER = 9973;

    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // use the thread index to get a seed
    unsigned int lcg_state = index * 352673253;

    for(int i = 0; i < num_accumulations; i++) {
        lcg_state = (LCG_MULT * lcg_state + LCG_ITER);

        int input_idx = lcg_state % (num_rows * feat_dim_in);
        int input_row = input_idx / feat_dim_in;
        int input_col = input_idx % feat_dim_in;
        float input_pmf = 1.0f / (num_rows * feat_dim_in);

        // now that we have an input col, we need to search that row in the weight in particular

        lcg_state = (LCG_MULT * lcg_state + LCG_ITER);
        int weight_idx = lcg_state % feat_dim_out;
        int weight_row = input_col; // column becomes row because of how matrix multiplication flips it
        int weight_col = weight_idx; // no need to modulate
        float weight_pmf = 1.0f / feat_dim_out;


        float input_val = input[input_idx]; // we can reuse this variable from before
        float weight_val = weight[weight_col * feat_dim_in + weight_row]; // need to flip because of how weight is represented

        // this actually should simplyfy down a lot
        atomicAdd(&output[input_row * feat_dim_out + weight_col], input_val * weight_val / (input_pmf * weight_pmf));
    }
}

torch::Tensor run_mc_gemm(torch::Tensor input, torch::Tensor weight) {
    std::cout << "Running regular Monte Carlo pass...\n";

    const int num_accumulations = 1;

    const int num_threads = 128;
    const int block_size = 128;

    cudaDeviceSynchronize();

    torch::Tensor output = torch::zeros({batch_size * seq_len, feat_dim_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();
    mc_gemm_kernel<<<num_threads / block_size, block_size>>>(
        batch_size * seq_len,
        feat_dim_in,
        feat_dim_out,
        num_accumulations,
        (float*)input.data_ptr(), 
        (float*)weight.data_ptr(), 
        (float*)output.data_ptr()
    );
    cudaDeviceSynchronize();

    output = output / (float)(num_threads * num_accumulations);
    return output;
}