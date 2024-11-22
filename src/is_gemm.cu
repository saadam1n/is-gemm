#include <cuda.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

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
        input = tensors.attr("input").toTensor().contiguous();
        weight = tensors.attr("weight").toTensor().contiguous(); 
    } catch (const c10::Error& e) {
        std::cout << "Error while loading input tensor file: " << e.what() << std::endl;
        return -1;
    }

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

    // now we need to preprocess the input and weight to create a sampling distribution
    // our goal is to sample accordingly to the inner product value
    // a way to approximate this: pick a random value in the input proportional to its value
    // then pick a value it can multiplty against proportional to its value
    // we should pick a value in the input just in case we have a 1xn input (so we have multiple options garaunteed)

    
    std::cout << "Creating input pmf.\n";
    // no idea if cumsum breaks contiguity so lets do it just to be safe
    torch::Tensor input_pmf = input.view(batch_size * seq_len * feat_dim_in).cumsum(0).contiguous();
    input_pmf /= input_pmf[batch_size * seq_len * feat_dim_in - 1].item<float>();


    std::cout << "Creating weight pmf.\n";
    // if we pick a specific value in the input, then that value can only be multiplied by values within a particular row
    // so the weight matrix needs to be summed up on the row dimension
    torch::Tensor weight_pmf = weight.cumsum(0);
    // we need to divide each row by the sum (which is in the last column)
    weight_pmf /= weight_pmf.narrow(0, feat_dim_out - 1, 1).expand({feat_dim_out, feat_dim_in});



    return 0;
}