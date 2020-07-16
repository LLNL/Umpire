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

const int n = 10;
size_t Scalar = 5;

void Copy(std::size_t** A, std::size_t** C){
    for(int i = 0; i < n; i++)
    {
        (*A)[i] = (*C)[i];
    }   
}

void Scale(std::size_t** B, std::size_t** C){
    for(int i = 0; i < n; i++)
    {
        (*B)[i] = (*C)[i] * Scalar;
    }   
}

void Add(std::size_t** A, std::size_t** B, std::size_t** C){
    for(int i = 0; i < n; i++)
    {
        (*C)[i] = (*A)[i] + (*B)[i];
    }   
}

void Triad(std::size_t** A, std::size_t** B, std::size_t** C){
    for(int i = 0; i < n; i++)
    {
        (*A)[i] = (*B)[i] + Scalar * (*C)[i];
    }   
}

void Allocate(umpire::Allocator alloc, std::size_t** A, std::size_t** B, std::size_t** C){

    *A = (std::size_t*) alloc.allocate(sizeof(size_t) * n);
    *B = (std::size_t*) alloc.allocate(sizeof(size_t) * n);
    *C = (std::size_t*) alloc.allocate(sizeof(size_t) * n);

    for (int i=0; i<n; i++) {
        (*A)[i] = (size_t) rand() % 100;
        (*B)[i] = (size_t) rand() % 100;
        (*C)[i] = (size_t) rand() % 100;
    }
}

void Deallocate(umpire::Allocator alloc, std::size_t** A, std::size_t** B, std::size_t** C){
    alloc.deallocate( *A );
    alloc.deallocate( *B );
    alloc.deallocate( *C );
}

void benchmark(std::string name){

    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc = rm.getAllocator(name);

    std::size_t* A = nullptr;
    std::size_t* B = nullptr;
    std::size_t* C = nullptr;
    
    Allocate(alloc,&A,&B,&C);
    auto begin_copy = std::chrono::system_clock::now();
    Copy(&A,&C);
    auto end_copy = std::chrono::system_clock::now();
    Deallocate(alloc,&A,&B,&C);

    Allocate(alloc,&A,&B,&C);
    auto begin_scale = std::chrono::system_clock::now();
    Scale(&B,&C);
    auto end_scale = std::chrono::system_clock::now();
    Deallocate(alloc,&A,&B,&C);

    Allocate(alloc,&A,&B,&C);
    auto begin_add = std::chrono::system_clock::now();
    Add(&A,&B,&C);
    auto end_add = std::chrono::system_clock::now();
    Deallocate(alloc,&A,&B,&C);

    Allocate(alloc,&A,&B,&C);
    auto begin_triad = std::chrono::system_clock::now();
    Triad(&A,&B,&C);
    auto end_triad = std::chrono::system_clock::now();
    Deallocate(alloc,&A,&B,&C);

    std::cout << name << std::endl;
    std::cout << "    Copy:     " <<  std::chrono::duration<double>(end_copy - begin_copy).count()/n << " sec/elements" <<std::endl;
    std::cout << "    Scale:    " << std::chrono::duration<double>(end_scale - begin_scale).count()/n << " sec/elements" << std::endl;
    std::cout << "    Add:      " <<  std::chrono::duration<double>(end_add - begin_add).count()/n << " sec/elements" << std::endl;
    std::cout << "    Triad:    " << std::chrono::duration<double>(end_triad - begin_triad).count()/n << " sec/elements" << std::endl;
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