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

const int interations{10000};
size_t Scalar = 5;

void Copy(std::size_t* A, std::size_t* C){
    for(int i = 0; i < interations; i++){
        A[i] = C[i];
    }   
}

void Scale(std::size_t* B, std::size_t* C){
    for(int i = 0; i < interations; i++){
        B[i] = C[i] * Scalar;
    }   
}

void Add(std::size_t* A, std::size_t* B, std::size_t* C){
    for(int i = 0; i < interations; i++){
        C[i] = A[i] + B[i];
    }   
}

void Triad(std::size_t* A, std::size_t* B, std::size_t* C){
    for(int i = 0; i < interations; i++){
        A[i] = B[i] + Scalar * C[i];
    }   
}

void Allocation_Initialized(umpire::Allocator alloc, std::size_t*& A, std::size_t*& B, std::size_t*& C){

    A = (std::size_t*) alloc.allocate(sizeof(size_t) * interations);
    B = (std::size_t*) alloc.allocate(sizeof(size_t) * interations);
    C = (std::size_t*) alloc.allocate(sizeof(size_t) * interations);

    for (int i=0; i<interations; i++) {
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

    std::cout << name << std::endl;
    std::cout << "    Copy:     " <<  std::chrono::duration<double>(end_copy - begin_copy).count()/interations << " sec/elements" <<std::endl;
    std::cout << "    Scale:    " << std::chrono::duration<double>(end_scale - begin_scale).count()/interations << " sec/elements" << std::endl;
    std::cout << "    Add:      " <<  std::chrono::duration<double>(end_add - begin_add).count()/interations << " sec/elements" << std::endl;
    std::cout << "    Triad:    " << std::chrono::duration<double>(end_triad - begin_triad).count()/interations << " sec/elements" << std::endl;
}

int main(int, char**) {
    benchmark("HOST");
    benchmark("FILE");
#if defined(UMPIRE_ENABLE_CUDA)
    benchmark("DEVICE");
    benchmark("UM");
#endif
#if defined(UMPIRE_ENABLE_HIP)
    benchmark("DEVICE");
#endif
}