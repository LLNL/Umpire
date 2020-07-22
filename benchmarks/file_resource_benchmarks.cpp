//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/FileMemoryResource.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <unistd.h>
#include <time.h>

int iterations;
size_t Scalar = 5;

void Copy(std::size_t* A, std::size_t* C){
    for(int i = 0; i < iterations; i++){
        A[i] = C[i];
    }   
}

void Scale(std::size_t* B, std::size_t* C){
    for(int i = 0; i < iterations; i++){
        B[i] = C[i] * Scalar;
    }   
}

void Add(std::size_t* A, std::size_t* B, std::size_t* C){
    for(int i = 0; i < iterations; i++){
        C[i] = A[i] + B[i];
    }   
}

void Triad(std::size_t* A, std::size_t* B, std::size_t* C){
    for(int i = 0; i < iterations; i++){
        A[i] = B[i] + Scalar * C[i];
    }   
}

void Allocation_Initialized(umpire::Allocator alloc, std::size_t*& A, std::size_t*& B, std::size_t*& C){

    A = (std::size_t*) alloc.allocate(sizeof(size_t) * iterations);
    B = (std::size_t*) alloc.allocate(sizeof(size_t) * iterations);
    C = (std::size_t*) alloc.allocate(sizeof(size_t) * iterations);

    for (int i=0; i<iterations; i++) {
        A[i] = (size_t) rand() % 100;
        B[i] = (size_t) rand() % 100;
        C[i] = (size_t) rand() % 100;
    }
}

void Deallocation_Requested(umpire::Allocator alloc, std::size_t* A, std::size_t* B, std::size_t* C){
    alloc.deallocate( A );
    alloc.deallocate( B );
    alloc.deallocate( C );
}

void benchmark(std::string name){

    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc = rm.getAllocator(name);

    std::size_t* A = nullptr;
    std::size_t* B = nullptr;
    std::size_t* C = nullptr;
    
    auto begin = std::chrono::system_clock::now();

    Allocation_Initialized(alloc,A,B,C);
    auto begin_copy = std::chrono::system_clock::now();
    Copy(A,C);
    auto end_copy = std::chrono::system_clock::now();
    Deallocation_Requested(alloc,A,B,C);

    Allocation_Initialized(alloc,A,B,C);
    auto begin_scale = std::chrono::system_clock::now();
    Scale(B,C);
    auto end_scale = std::chrono::system_clock::now();
    Deallocation_Requested(alloc,A,B,C);

    Allocation_Initialized(alloc,A,B,C);
    auto begin_add = std::chrono::system_clock::now();
    Add(A,B,C);
    auto end_add = std::chrono::system_clock::now();
    Deallocation_Requested(alloc,A,B,C);

    Allocation_Initialized(alloc,A,B,C);
    auto begin_triad = std::chrono::system_clock::now();
    Triad(A,B,C);
    auto end_triad = std::chrono::system_clock::now();
    Deallocation_Requested(alloc,A,B,C);

    auto end = std::chrono::system_clock::now();

    std::cout << name << std::endl;
    std::cout << "    Copy:     " << ((2 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_copy - begin_copy).count() << " MB/sec" <<std::endl;
    std::cout << "    Scale:    " << ((2 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_scale - begin_scale).count() << " MB/sec" << std::endl;
    std::cout << "    Add:      " << ((3 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_add - begin_add).count() << " MB/sec" << std::endl;
    std::cout << "    Triad:    " << ((3 * sizeof(size_t) * iterations) * 1.0E-6)/std::chrono::duration<double>(end_triad - begin_triad).count() << " MB/sec" << std::endl;
    std::cout << "    Total:     " << std::chrono::duration<double>(end - begin).count() << " sec" <<std::endl;
}

int main(int, char** argv) {
    iterations = atoi(argv[1]);;
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