#pragma once

#include <cuda.h>
#include <torch/script.h>
#include <iostream>

#include "common.hpp"

__global__ void is_gemm_kernel(
    int num_rows, int feat_dim_in, int feat_dim_out, int num_accumulations, int seed,
    float* input, float* input_cmf,
    float* weight, float* weight_cmf,
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
        float selected_input_pmf = hybrid_taus(state);

        int input_idx = binary_search(input_cmf, num_rows * feat_dim_in, selected_input_pmf);
        int input_row = input_idx / feat_dim_in;
        int input_col = input_idx % feat_dim_in;
        float input_pmf = input_cmf[input_idx] - (input_idx != 0 ? input_cmf[input_idx - 1] : 0.0);

        // now that we have an input col, we need to search that row in the weight in particular

        float selected_weight_pmf = hybrid_taus(state);
        int weight_row = input_col; // column becomes row because of how matrix multiplication flips it
        int weight_idx = binary_search(&weight_cmf[weight_row * feat_dim_out], feat_dim_out, selected_weight_pmf);
        int weight_col = weight_idx; // no need to modulate
        float weight_pmf =  weight_cmf[weight_row * feat_dim_out + weight_col] - (weight_col != 0 ? weight_cmf[weight_row * feat_dim_out + weight_col - 1] : 0.0);


        float input_val = input[input_idx]; // we can reuse this variable from before
        float weight_val = weight[weight_col * feat_dim_in + weight_row]; // need to flip because of how weight is represented

        // this actually should simplyfy down a lot
        atomicAdd(&output[input_row * feat_dim_out + weight_col], input_val * weight_val / (input_pmf * weight_pmf));
    }
}

torch::Tensor run_is_gemm(torch::Tensor input, torch::Tensor weight) {
    // now we need to preprocess the input and weight to create a sampling distribution
    // our goal is to sample accordingly to the inner product value
    // a way to approximate this: pick a random value in the input proportional to its value
    // then pick a value it can multiplty against proportional to its value
    // we should pick a value in the input just in case we have a 1xn input (so we have multiple options garaunteed)


    std::cout << "Creating input pmf...\n";
    // no idea if cumsum breaks contiguity so lets do it just to be safe
    torch::Tensor input_cmf = input.view(batch_size * seq_len * feat_dim_in).abs().cumsum(0).contiguous();
    float matrix_norm = input_cmf[batch_size * seq_len * feat_dim_in - 1].item<float>();
    input_cmf /= matrix_norm;

    std::cout << "Creating weight pmf...\n";
    // if we pick a specific value in the input, then that value can only be multiplied by values within a particular row
    // so the weight matrix needs to be summed up on the row dimension
    torch::Tensor weight_cmf = weight.abs().cumsum(0);

    // we need to divide each row by the sum (which is in the last row)
    torch::Tensor row_norm = weight_cmf.narrow(0, feat_dim_out - 1, 1).expand({feat_dim_out, feat_dim_in});
    weight_cmf = weight_cmf / row_norm;
    weight_cmf = weight_cmf.transpose(0, 1).contiguous(); // make this easier for our code to read


    // now that are tensors are set up, let's do the actual accumulation
    const int num_accumulations = 1;
    const int num_threads = 128;
    const int block_size = 128;

    std::cout << "Running importance sampling pass...\n";

    torch::Tensor output = torch::zeros({batch_size * seq_len, feat_dim_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();

    // sync just in case pytorch does stuff differently
    cudaDeviceSynchronize();
    is_gemm_kernel<<<num_threads / block_size, block_size>>>(
        batch_size * seq_len,
        feat_dim_in,
        feat_dim_out,
        num_accumulations,
        time(nullptr),
        (float*)input.data_ptr(), 
        (float*)input_cmf.data_ptr(), 
        (float*)weight.data_ptr(), 
        (float*)weight_cmf.data_ptr(), 
        (float*)output.data_ptr()
    );
    cudaDeviceSynchronize();

    output = output / (float)(num_threads * num_accumulations);

    return output;
}