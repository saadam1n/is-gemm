# Importance Sampling GEMM (IS-GEMM)

This GEMM method for sparce matricies is a monte carlo method that utilizes importance sampling to focus more computation towards inner product terms that are more likely to have an impact on the output.

## Building and Running

This project can utilize the same `.pt` files from the `rtx-attention` project. 

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<path to libtorch> ../
make -j 16
./bin/is_gemm <path to tensor file>
```