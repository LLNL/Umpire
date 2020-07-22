//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/FileMemoryResource.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <omp.h>

#include <unistd.h>
#include <time.h>

int iterations;
size_t Scalar = 5;



void Copy(std::size_t* A, std::size_t* C){
    #pragma omp parallel for
        for(int i = 0; i < iterations; i++){
            A[i] = C[i];
        }   
}

void Scale(std::size_t* B, std::size_t* C){
    #pragma omp parallel for
        for(int i = 0; i < iterations; i++){
            B[i] = C[i] * Scalar;
        }   
}

void Add(std::size_t* A, std::size_t* B, std::size_t* C){
    #pragma omp parallel for
        for(int i = 0; i < iterations; i++){
            C[i] = A[i] + B[i];
        }   
}

void Triad(std::size_t* A, std::size_t* B, std::size_t* C){
    #pragma omp parallel for
        for(int i = 0; i < iterations; i++){
            A[i] = B[i] + Scalar * C[i];
        }   
}

void* Allocation(umpire::Allocator alloc){
    return alloc.allocate(sizeof(size_t) * iterations);
}

void Initialized(umpire::Allocator alloc, std::size_t* A){

    #pragma omp parallel for
        for (int i=0; i<iterations; i++) {
            A[i] = (size_t) rand() % 100;
        }
}

void benchmark(std::string name){

    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc = rm.getAllocator(name);

    std::size_t* A = nullptr;
    std::size_t* B = nullptr;
    std::size_t* C = nullptr;

    auto begin = std::chrono::system_clock::now();

    auto begin_copy_plus = std::chrono::system_clock::now();
    A = (std::size_t*) Allocation(alloc);
    C = (std::size_t*) Allocation(alloc);
    Initialized(alloc,A);
    Initialized(alloc,C);
    auto begin_copy = std::chrono::system_clock::now();
    Copy(A,C);
    auto end_copy = std::chrono::system_clock::now();
    alloc.deallocate(A);
    alloc.deallocate(C);
    auto end_copy_plus = std::chrono::system_clock::now();

    auto begin_scale_plus = std::chrono::system_clock::now();
    B = (std::size_t*) Allocation(alloc);
    C = (std::size_t*) Allocation(alloc);
    Initialized(alloc,B);
    Initialized(alloc,C);
    auto begin_scale = std::chrono::system_clock::now();
    Scale(B,C);
    auto end_scale = std::chrono::system_clock::now();
    alloc.deallocate(B);
    alloc.deallocate(C);
    auto end_scale_plus = std::chrono::system_clock::now();

    auto begin_add_plus = std::chrono::system_clock::now();
    A = (std::size_t*) Allocation(alloc);
    B = (std::size_t*) Allocation(alloc);
    C = (std::size_t*) Allocation(alloc);
    Initialized(alloc,A);
    Initialized(alloc,B);
    Initialized(alloc,C);
    auto begin_add = std::chrono::system_clock::now();
    Add(A,B,C);
    auto end_add = std::chrono::system_clock::now();
    alloc.deallocate(A);
    alloc.deallocate(B);
    alloc.deallocate(C);
    auto end_add_plus = std::chrono::system_clock::now();

    auto begin_triad_plus = std::chrono::system_clock::now();
    A = (std::size_t*) Allocation(alloc);
    B = (std::size_t*) Allocation(alloc);
    C = (std::size_t*) Allocation(alloc);
    Initialized(alloc,A);
    Initialized(alloc,B);
    Initialized(alloc,C);
    auto begin_triad = std::chrono::system_clock::now();
    Triad(A,B,C);
    auto end_triad = std::chrono::system_clock::now();
    alloc.deallocate(A);
    alloc.deallocate(B);
    alloc.deallocate(C);
    auto end_triad_plus = std::chrono::system_clock::now();

    auto end = std::chrono::system_clock::now();

    std::cout << name << std::endl;
    std::cout << "  Copy:       " << ((2 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_copy - begin_copy).count() << " MB/sec" <<std::endl;
    std::cout << "  Copy Time:  " << std::chrono::duration<double>(end_copy_plus - begin_copy_plus).count() << " sec" <<std::endl;
    std::cout << "  ---------------------------------------\n";
    std::cout << "  Scale:      " << ((2 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_scale - begin_scale).count() << " MB/sec" << std::endl;
    std::cout << "  Scale Time: " << std::chrono::duration<double>(end_scale_plus - begin_scale_plus).count() << " sec" <<std::endl;
    std::cout << "  ---------------------------------------\n";
    std::cout << "  Add:        " << ((3 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_add - begin_add).count() << " MB/sec" << std::endl;
    std::cout << "  Add Time:   " << std::chrono::duration<double>(end_add_plus - begin_add_plus).count() << " sec" <<std::endl;
    std::cout << "  ---------------------------------------\n";
    std::cout << "  Triad:      " << ((3 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_triad - begin_triad).count() << " MB/sec" << std::endl;
    std::cout << "  Triad Time: " << std::chrono::duration<double>(end_triad_plus - begin_triad_plus).count() << " sec" <<std::endl;
    std::cout << "  ---------------------------------------\n";
    std::cout << "  Total Time: " << std::chrono::duration<double>(end - begin).count() << " sec" <<std::endl<<std::endl;
}

int main(int, char** argv) {
    iterations = atoi(argv[1]);
    std::cout << "Array Size: " << iterations << std::endl;
    benchmark("HOST");
    benchmark("FILE");
#if defined(UMPIRE_ENABLE_CUDA)
    benchmark("UM");
#endif
#if defined(UMPIRE_ENABLE_HIP)
    benchmark("DEVICE");
#endif
}