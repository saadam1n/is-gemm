#include <cuda.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

// a bunch of random primes
#define LCG_MAX 1000000007
#define LCG_MULT 999983 
#define LCG_ITER 9973

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

__global__ void is_gemm(
    int num_rows, int feat_dim_in, int feat_dim_out, int num_accumulations, 
    float* input, float* input_cmf, float* weight, float* weight_cmf, 
    float* output, unsigned int* sample_cnt
) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    // some random number to get the seed
    unsigned int lcg_state = index * 352673253;

    for(int i = 0; i < num_accumulations; i++) {
        lcg_state = (LCG_MULT * lcg_state + LCG_ITER) % LCG_MAX;

        int input_idx = binary_search(input_cmf, num_rows * feat_dim_in, (float)lcg_state / LCG_MAX);
        int input_row = input_idx / feat_dim_in;
        int input_col = input_idx % feat_dim_in;
        float input_pmf = input_cmf[input_idx] - (input_idx != 0 ? input_cmf[input_idx - 1] : 0.0);

        // now that we have an input col, we need to search that row in the weight in particular

        lcg_state = (LCG_MULT * lcg_state + LCG_ITER) % LCG_MAX;
        int weight_row = input_col; // column becomes row because of how matrix multiplication flips it
        int weight_idx = binary_search(&weight_cmf[weight_row * feat_dim_out], feat_dim_out, (float)lcg_state / LCG_MAX);
        int weight_col = weight_idx; // no need to modulate
        float weight_pmf = weight_cmf[weight_row * feat_dim_out + weight_col] - (weight_col != 0 ? weight_cmf[weight_row * feat_dim_out + weight_col - 1] : 0.0);


        float input_val = input[input_idx]; // we can reuse this variable from before
        float weight_val = weight[weight_col * feat_dim_in + weight_row]; // need to flip because of how weight is represented


        // this actually should simplyfy down a lot
        atomicAdd(&output[input_row * feat_dim_out + weight_col], input_val * weight_val / (input_pmf * weight_pmf));
        atomicAdd(&sample_cnt[input_row * feat_dim_out + weight_col], 1);
    }
}

__global__ void mc_gemm(
    int num_rows, int feat_dim_in, int feat_dim_out, int num_accumulations, 
    float* input, float* weight,
    float* output, unsigned int* sample_cnt, unsigned int* stuff
) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    // some random number to get the seed
    unsigned int lcg_state = index * 352673253;

    for(int i = 0; i < num_accumulations; i++) {
        lcg_state = (LCG_MULT * lcg_state + LCG_ITER) % LCG_MAX;

        int input_idx = lcg_state % (num_rows * feat_dim_in);
        int input_row = input_idx / feat_dim_in;
        int input_col = input_idx % feat_dim_in;
        float input_pmf = 1.0f / (num_rows * feat_dim_in);

        // now that we have an input col, we need to search that row in the weight in particular

        lcg_state = (LCG_MULT * lcg_state + LCG_ITER) % LCG_MAX;
        int weight_idx = lcg_state % feat_dim_out;
        int weight_row = input_col; // column becomes row because of how matrix multiplication flips it
        int weight_col = weight_idx; // no need to modulate
        float weight_pmf = 1.0f / feat_dim_out;


        float input_val = input[input_idx]; // we can reuse this variable from before
        float weight_val = weight[weight_col * feat_dim_in + weight_row]; // need to flip because of how weight is represented

        if(input_row == 1 && weight_col == 1) {
            atomicAdd(&stuff[input_col], 1);
            stuff[input_col + 4] = input_val * weight_val;
            stuff[input_col + 8] = input_val;
            stuff[input_col + 12] = weight_val;
        }

        // this actually should simplyfy down a lot
        atomicAdd(&output[input_row * feat_dim_out + weight_col], input_val * weight_val / (input_pmf * weight_pmf));
        atomicAdd(&sample_cnt[input_row * feat_dim_out + weight_col], 1);
    }
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

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cout << "Usage: ./is-gemm <path to tensors file>" << std::endl;
        return -1;
    }

    std::string tensors_file = argv[1];
    std::cout << "Loading tensors from " << tensors_file << std::endl;

    torch::Tensor input, weight;
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

    float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    input = torch::from_blob(arr, {1, 3, 4}, torch::dtype(torch::kFloat32)).contiguous().to(torch::kCUDA);
    weight = torch::from_blob(arr, {3, 4}, torch::dtype(torch::kFloat32)).contiguous().to(torch::kCUDA);

    unsigned int batch_size, seq_len, feat_dim_in, feat_dim_out, w_feat_dim_in;

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

    std::cout << "Calculating reference matrix...\n";
    torch::Tensor reference = torch::matmul(input, weight.transpose(0, 1));

    // now we need to preprocess the input and weight to create a sampling distribution
    // our goal is to sample accordingly to the inner product value
    // a way to approximate this: pick a random value in the input proportional to its value
    // then pick a value it can multiplty against proportional to its value
    // we should pick a value in the input just in case we have a 1xn input (so we have multiple options garaunteed)

    
    std::cout << "Creating input pmf...\n";
    // no idea if cumsum breaks contiguity so lets do it just to be safe
    torch::Tensor input_cmf = input.view(batch_size * seq_len * feat_dim_in).abs().cumsum(0).contiguous();
    input_cmf /= input_cmf[batch_size * seq_len * feat_dim_in - 1].item<float>();


    std::cout << "Creating weight pmf...\n";
    // if we pick a specific value in the input, then that value can only be multiplied by values within a particular row
    // so the weight matrix needs to be summed up on the row dimension
    torch::Tensor weight_cmf = weight.abs().cumsum(0);

    // we need to divide each row by the sum (which is in the last row)
    weight_cmf /= weight_cmf.narrow(0, feat_dim_out - 1, 1).expand({feat_dim_out, feat_dim_in});
    weight_cmf = weight_cmf.transpose(0, 1).contiguous(); // make this easier for our code to read

    std::cout << "Creating output and sample count matrices...\n";
    torch::Tensor output = torch::zeros({batch_size * seq_len, feat_dim_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)).contiguous();
    torch::Tensor sample_cnt = torch::zeros({batch_size * seq_len, feat_dim_out}, torch::dtype(torch::kUInt32).device(torch::kCUDA, 0)).contiguous();

    torch::Tensor stuff = torch::zeros({4, 4}, torch::dtype(torch::kUInt32).device(torch::kCUDA, 0)).contiguous();

    #if 1
    // now that are tensors are set up, let's do the actual accumulation
    // since I'm very lazy I will use a LCG
    // each thread will have its own LCG state that is initialized from it's seed index
    const int num_accumulations = 32;

    std::cout << "Running importance sampling pass...\n";
    const int num_threads_is = 128;
    const int block_size_is = 128;
    #if 1
    is_gemm<<<num_threads_is / block_size_is, block_size_is>>>(
        batch_size * seq_len,
        feat_dim_in,
        feat_dim_out,
        num_accumulations,
        (float*)input.data_ptr(), 
        (float*)input_cmf.data_ptr(), 
        (float*)weight.data_ptr(), 
        (float*)weight_cmf.data_ptr(), 
        (float*)output.data_ptr(),
        (unsigned int*) sample_cnt.data_ptr()
    );
    #else
    mc_gemm<<<num_threads_is / block_size_is, block_size_is>>>(
        batch_size * seq_len,
        feat_dim_in,
        feat_dim_out,
        num_accumulations,
        (float*)input.data_ptr(), 
        (float*)weight.data_ptr(), 
        (float*)output.data_ptr(),
        (unsigned int*) sample_cnt.data_ptr(),
        (unsigned int*) stuff.data_ptr()
    );
    #endif
    cudaDeviceSynchronize();

    std::cout << "Running division pass...\n";
    output /= (float)(num_threads_is * num_accumulations);
    #else
    dim3 block_size = {32, 32, 1};
    dim3 grid_size = {(batch_size * seq_len - 1) / 32 + 1, (feat_dim_out - 1) / 32 + 1, 1};
    brute_force_gemm<<<grid_size, block_size>>>(
        batch_size * seq_len,
        feat_dim_in,
        feat_dim_out,
        (float*)input.data_ptr(), 
        (float*)weight.data_ptr(), 
        (float*)output.data_ptr()
    );
    #endif

    torch::Tensor l1_error = torch::abs(output - reference);
    std::cout << "Average L1 error was " << l1_error.sum().item<float>() / (batch_size * seq_len * feat_dim_out) << "\n";
    std::cout << "Max possible L1 error is " <<  torch::abs(reference).sum().item<float>() / (batch_size * seq_len * feat_dim_out) << "\n";

    std::cout << input << std::endl;
    std::cout << weight << std::endl;
    std::cout << output << std::endl;
    std::cout << sample_cnt << std::endl;
    std::cout << reference << std::endl;
    std::cout << stuff << std::endl;

    std::cout << "CMFs" << std::endl;
    std::cout << input_cmf << std::endl;
    std::cout << weight_cmf << std::endl;

    return 0;
}