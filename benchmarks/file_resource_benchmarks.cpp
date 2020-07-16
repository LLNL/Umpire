//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/FileMemoryResource.hpp"

#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

const int n = 10;
size_t Scalar = 5;

auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
std::size_t* A_pointer;
std::size_t* B_pointer;
std::size_t* C_pointer;

std::size_t* A;
std::size_t* B;
std::size_t* C;

void Copy(){
    for(int i = 0; i < n; i++)
    {
        A[i] = C[i];
    }   
}

void Scale(){
    for(int i = 0; i < n; i++)
    {
        B[i] = C[i] * Scalar;
    }   
}

void Add(){
    for(int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }   
}

void Triad(){
    for(int i = 0; i < n; i++)
    {
        A[i] = B[i] + Scalar * C[i];
    }   
}

void Allocate(){
    A_pointer = (std::size_t*) alloc->allocate(sysconf(_SC_PAGE_SIZE) * (sizeof(size_t) * 3 * n) );
    B_pointer = (std::size_t*) alloc->allocate(sysconf(_SC_PAGE_SIZE) * (sizeof(size_t) * 3 * n) );
    C_pointer = (std::size_t*) alloc->allocate(sysconf(_SC_PAGE_SIZE) * (sizeof(size_t) * 3 * n) );

    A = A_pointer;
    B = B_pointer;
    C = C_pointer;

    A = new size_t[n];
    B = new size_t[n];
    C = new size_t[n];

    for (int i=0; i<n; i++) {
        A[i] = (size_t) rand() % 100;
        B[i] = (size_t) rand() % 100;
        C[i] = (size_t) rand() % 100;
    }
}

void Deallocate(){
    alloc->deallocate( A_pointer );
    alloc->deallocate( B_pointer );
    alloc->deallocate( C_pointer );
}

int main(int, char**) {

    Allocate();
    auto begin_copy = std::chrono::system_clock::now();
    Copy();
    auto end_copy = std::chrono::system_clock::now();
    Deallocate();

    Allocate();
    auto begin_scale = std::chrono::system_clock::now();
    Scale();
    auto end_scale = std::chrono::system_clock::now();
    Deallocate();

    Allocate();
    auto begin_add = std::chrono::system_clock::now();
    Add();
    auto end_add = std::chrono::system_clock::now();
    Deallocate();

    Allocate();
    auto begin_triad = std::chrono::system_clock::now();
    Triad();
    auto end_triad = std::chrono::system_clock::now();
    Deallocate();

    std::cout << "    Copy:     " <<  std::chrono::duration<double>(end_copy - begin_copy).count()/1 << std::endl;
    std::cout << "    Scale:    " << std::chrono::duration<double>(end_scale - begin_scale).count()/1 << std::endl;
    std::cout << "    Add:      " <<  std::chrono::duration<double>(end_add - begin_add).count()/1 << std::endl;
    std::cout << "    Triad:    " << std::chrono::duration<double>(end_triad - begin_triad).count()/1 << std::endl; 
    
    return 0;
}