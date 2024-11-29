#include <cuda.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

// a bunch of random primes
#define LCG_MULT 999983 
#define LCG_ITER 9973

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




__global__ void is_gemm_kernel(
    int num_rows, int feat_dim_in, int feat_dim_out, int num_accumulations, int seed,
    float* input, float* input_cmf,
    float* weight, float* weight_cmf,
    float* output
) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // set up the a PRNG state
    unsigned int thread_seed = index + seed;
    uint4 state = {thread_seed * 3262432, thread_seed * 5747547, thread_seed * 325325, thread_seed * 3252626};

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

    const int num_threads = 24;
    const int block_size = 24;

    // sync just in case pytorch does stuff differently
    cudaDeviceSynchronize();

    std::cout << "Running importance sampling pass...\n";
    torch::Tensor output = torch::zeros({batch_size * seq_len, feat_dim_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();
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

    output = output / (float)(num_threads * num_accumulations);

    return output;
}

__global__ void mc_gemm_kernel(
    int num_rows, int feat_dim_in, int feat_dim_out, int num_accumulations, 
    float* input, float* weight,
    float* output
) {
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

__global__ void brute_force_gemm(
    int num_rows, int feat_dim_in, int feat_dim_out, 
    float* input, float* weight,
    float* output
) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int col = threadIdx.y + blockIdx.y * blockDim.y;

    if(row >= num_rows || col >= feat_dim_out) {
        return;
    }

    float sum = 0.0f;
    for(int i = 0; i < feat_dim_in; i++) {
        sum += input[row * feat_dim_in + i] * weight[col * feat_dim_in + i];
    }

    output[row * feat_dim_out + col] = sum;
}

//#define USE_EXTERNAL_MATRICES

int main(int argc, char** argv) {


    // ============= Start of setup code =============

    // load inputs and weights from file if USE_EXTERNAL_MATRICES is true
    // otherwise we use some pre-defined matrices
    torch::Tensor input, weight;

    #ifdef USE_EXTERNAL_MATRICES
    if(argc != 2) {
        std::cout << "Usage: ./is-gemm <path to tensors file>" << std::endl;
        return -1;
    }

    std::string tensors_file = argv[1];
    std::cout << "Loading tensors from " << tensors_file << std::endl;

    try {
        torch::jit::script::Module tensors = torch::jit::load(tensors_file);
        // For GEMM(A, B), we assume A as input and B as weight 
        // We do .contiguous() to ensure the memory layout is as expected
        input = tensors.attr("input").toTensor().to(torch::kFloat32).contiguous();
        weight = tensors.attr("weight").toTensor().to(torch::kFloat32).contiguous(); 
    } catch (const c10::Error& e) {
        std::cout << "Error while loading input tensor file: " << e.what() << std::endl;
        return -1;
    }
    
    #else
    std::cout << "Using pre-defined matrices for testing GEMM.\n";

    if(argc == 2) {
        std::cout << "Warning: ignoring second argument \"" << argv[1] << "\" in this mode.\n"  
                  << "Please recompile with USE_EXTERNAL_MATRICES defined in order to use external input and weight matrices."
                  << std::endl;
    }

    float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    input = torch::from_blob(arr, {1, 3, 4}, torch::dtype(torch::kFloat32)).contiguous().to(torch::kCUDA);
    weight = torch::from_blob(arr, {3, 4}, torch::dtype(torch::kFloat32)).contiguous().to(torch::kCUDA);
    #endif



    // Assume input shape as (batch_size, seq_len, feat_dim_in)
    batch_size = static_cast<unsigned int>(input.size(0));
    seq_len = static_cast<unsigned int>(input.size(1));
    feat_dim_in = static_cast<unsigned int>(input.size(2));
    // Reshape input shape as (batch_size * seq_len, feat_dim_in) to make GEMM easier
    input = input.reshape({batch_size * seq_len, feat_dim_in}).contiguous(); 

    // Assume weight shape as (feat_dim_out, feat_dim_in)
    feat_dim_out = static_cast<unsigned int>(weight.size(0));
    w_feat_dim_in = static_cast<unsigned int>(weight.size(1));

    std::cout << "Tensors successfully loaded.\n";
    std::cout << "\t(I) Batch size:   " << batch_size << "\n";
    std::cout << "\t(I) Sequence len: " << seq_len << "\n";
    std::cout << "\t(I) Total rows: " << batch_size * seq_len << "\n";
    std::cout << "\t(I) Feat dim in: " << feat_dim_in << "\n";
    std::cout << "\t(W) Feat dim out: " << feat_dim_out << "\n";
    std::cout << "\t(W) Feat dim in: " << w_feat_dim_in << "\n";

    // ============= End of setup code =============



    // ============= Start of testing code =============

    
    // use a GEMM algorithm to calculate the output matrix
    torch::Tensor output = run_is_gemm(input, weight);


    std::cout << "Calculating reference matrix...\n";
    torch::Tensor reference = torch::matmul(input, weight.transpose(0, 1));

    torch::Tensor l1_error = torch::abs(output - reference);

    const int MAX_ELEMENTS_TO_DISPLAY = 1024;
    if(output.numel() < MAX_ELEMENTS_TO_DISPLAY) {
        std::cout << "==== MATRICES ====\n";
        std::cout << "Reference:\n";
        std::cout << reference << "\n";
        std::cout << "Output:\n";
        std::cout << output << "\n";
        std::cout << "L1 error:\n";
        std::cout << l1_error << "\n";
    } else {
        std::cout << "Matrices not shown because total number of elements in each matrix is greater than " << MAX_ELEMENTS_TO_DISPLAY << "\n";
    }
    std::cout << "\n\n";

    std::cout << "==== RESULTS ====\n";
    std::cout << "Average L1 error was " << l1_error.sum().item<float>() / (batch_size * seq_len * feat_dim_out) << "\n";
    std::cout << "Max possible L1 error should be " <<  torch::abs(reference).sum().item<float>() / (batch_size * seq_len * feat_dim_out) << "\n";
    std::cout << "\n\n";

    // ============= End of testing code =============

    return 0;
}