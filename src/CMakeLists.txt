find_package(Torch REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(is_gemm 
    common.hpp
    gemm_test.cu
    is_gemm.hpp
    mc_gemm.hpp
    ais_gemm.hpp
)

target_link_libraries(is_gemm
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
)

target_include_directories(is_gemm PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET is_gemm PROPERTY CXX_STANDARD 17)

target_compile_options(
    is_gemm 
    PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>: 
        --generate-line-info
        --source-in-ptx
    >
)