#include <cuda.h>
#include <torch/script.h>
#include <iostream>

#include "common.hpp"
#include "is_gemm.hpp"
#include "mc_gemm.hpp"
#include "ais_gemm.hpp"

#define USE_EXTERNAL_MATRICES

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
    torch::Tensor output = run_ais_gemm(input, weight);


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