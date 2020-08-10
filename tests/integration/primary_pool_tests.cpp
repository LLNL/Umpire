//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "camp/camp.hpp"

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/QuickPool.hpp"

#include <random>
#include <string>
#include <vector>

template <typename T>
struct tag_to_string
{
};

template<>
struct tag_to_string<umpire::strategy::DynamicPoolList>
{
    static constexpr const char* value = "DynamicPoolList";
};

template<>
struct tag_to_string<umpire::strategy::DynamicPoolMap>
{
    static constexpr const char* value = "DynamicPoolMap";
};

template<>
struct tag_to_string<umpire::strategy::QuickPool>
{
    static constexpr const char* value = "QuickPool";
};

struct host_resource_tag
{
};

template<>
struct tag_to_string<host_resource_tag>
{
    static constexpr const char* value = "HOST";
};

#if defined(UMPIRE_ENABLE_DEVICE)
struct device_resource_tag
{
};

template<>
struct tag_to_string<device_resource_tag>
{
    static constexpr const char* value = "DEVICE";
};
#endif

#if defined(UMPIRE_ENABLE_UM)
struct um_resource_tag
{
};

template<>
struct tag_to_string<um_resource_tag>
{
    static constexpr const char* value = "UM";
};
#endif

#if defined(UMPIRE_ENABLE_PINNED)
struct pinned_resource_tag
{
};

template<>
struct tag_to_string<pinned_resource_tag>
{
    static constexpr const char* value = "PINNED";
};
#endif

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
struct file_resource_tag
{
};

template<>
struct tag_to_string<file_resource_tag>
{
    static constexpr const char* value = "FILE";
};
#endif

using ResourceTypes = camp::list<
                          host_resource_tag
#if defined(UMPIRE_ENABLE_DEVICE)
                        , device_resource_tag
#endif
#if defined(UMPIRE_ENABLE_UM)
                        , um_resource_tag
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                        , pinned_resource_tag
#endif
#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
                        , file_resource_tag
#endif
                        >;

using PoolTypes = camp::list<umpire::strategy::DynamicPoolList,
                    umpire::strategy::DynamicPoolMap,
                    umpire::strategy::QuickPool>;

using TestTypes = camp::cartesian_product<PoolTypes, ResourceTypes>;

//
// Unroll types for gtest testing::Types
//
template <class T>
struct Test;
template <class... T>
struct Test<camp::list<T...>>
{
  using Types = ::testing::Types<T...>;
};

using PoolTestTypes = Test<TestTypes>::Types;

template<typename PoolTuple>
class PrimaryPoolTest : public ::testing::Test
{
    public:
        using Pool = typename camp::at<PoolTuple, camp::num<0>>::type;
        using ResourceType = typename camp::at<PoolTuple, camp::num<1>>::type;

        void SetUp() override
        {
            static int unique_counter{0};
            auto& rm = umpire::ResourceManager::getInstance();
            m_resource_name = std::string( tag_to_string<ResourceType>::value );

            std::string name{ std::string{"pool_test"}
                + std::string{"_"} + std::string{tag_to_string<Pool>::value}
                + std::string{"_"} + std::string{m_resource_name}
                + std::string{"_"} + std::to_string(unique_counter++) };

            m_allocator = new umpire::Allocator(
                            rm.makeAllocator<Pool>(name
                                            , rm.getAllocator(m_resource_name)
                                            , m_initial_pool_size
                                            , m_min_pool_growth_size
                                            , m_alignment));
        }

        void TearDown() override
        {
            delete m_allocator;
            m_allocator = nullptr;
        }

        umpire::Allocator* m_allocator;
        const std::size_t m_big{512};
        const std::size_t m_nothing{0};
        const std::size_t m_initial_pool_size{ 16*1024 };
        const std::size_t m_min_pool_growth_size{ 64 };
        const std::size_t m_alignment{ 256 };
        std::string m_resource_name;
};

TYPED_TEST_SUITE(PrimaryPoolTest, PoolTestTypes,);

TYPED_TEST(PrimaryPoolTest, AllocateDeallocateBig)
{
    double* data = static_cast<double*>(
        this->m_allocator->allocate(this->m_big*sizeof(double)));

    ASSERT_NE(nullptr, data);

    this->m_allocator->deallocate(data);
}

TYPED_TEST(PrimaryPoolTest, Allocate)
{
    void* data = nullptr;
    data = this->m_allocator->allocate(100);
    this->m_allocator->deallocate(data);
}

TYPED_TEST(PrimaryPoolTest, Sizes)
{
    void* data{nullptr};
    const std::size_t size{this->m_initial_pool_size-1};

    ASSERT_NO_THROW(
        {
            data = this->m_allocator->allocate(size);
        });

    ASSERT_EQ(this->m_allocator->getSize(data), size);
    ASSERT_GE(this->m_allocator->getCurrentSize(), size);
    ASSERT_EQ(this->m_allocator->getHighWatermark(), size);
    ASSERT_EQ(this->m_allocator->getActualSize(), this->m_initial_pool_size);

    void* data2{nullptr};

    ASSERT_NO_THROW(
        {
            data2 = this->m_allocator->allocate(this->m_initial_pool_size);
        });
    ASSERT_NO_THROW(
        {
            this->m_allocator->deallocate(data);
        });

    ASSERT_GE(this->m_allocator->getCurrentSize(), this->m_initial_pool_size);

    ASSERT_EQ(this->m_allocator->getHighWatermark(),
                this->m_initial_pool_size+size);

    ASSERT_GE(this->m_allocator->getActualSize(),
                this->m_initial_pool_size+this->m_min_pool_growth_size);

    ASSERT_EQ(this->m_allocator->getSize(data2), this->m_initial_pool_size);

    ASSERT_NO_THROW({ this->m_allocator->deallocate(data2); });
}

TYPED_TEST(PrimaryPoolTest, Duplicate)
{
    using Pool  = typename TestFixture::Pool;
    auto& rm = umpire::ResourceManager::getInstance();

    ASSERT_TRUE( rm.isAllocator(this->m_allocator->getName()) );

    ASSERT_ANY_THROW( rm.makeAllocator<Pool>(
                        this->m_allocator->getName(),
                        rm.getAllocator(this->m_resource_name)));
}

TYPED_TEST(PrimaryPoolTest, Alignment)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::size_t> dist(1, this->m_big);
    std::vector<void*> allocations;
    const int num_allocations{1000};

    for (int i=0; i<num_allocations; ++i) {
        const std::size_t size{dist(mt)};
        void *ptr{nullptr};

        ASSERT_NO_THROW({ ptr = this->m_allocator->allocate(size); });

        EXPECT_TRUE(
            0 == (reinterpret_cast<std::ptrdiff_t>(ptr) % this->m_alignment))
            << "Allocation for size: " << size << " : "
            << ptr << " mod " << this->m_alignment << " = "
            << (reinterpret_cast<std::ptrdiff_t>(ptr) % this->m_alignment)
            << std::endl;

        ASSERT_TRUE(
            0 == (reinterpret_cast<std::ptrdiff_t>(ptr) % this->m_alignment));

        allocations.push_back(ptr);
    }

    for (auto alloc : allocations) {
        ASSERT_NO_THROW(
            {
                this->m_allocator->deallocate(alloc);
            });
    }
}

TYPED_TEST(PrimaryPoolTest, Works)
{
    void* ptr_one{nullptr};
    void* ptr_two{nullptr};

    ASSERT_NO_THROW(
        {
            ptr_one = this->m_allocator->allocate(62);
            ptr_two = this->m_allocator->allocate(1024);
            this->m_allocator->deallocate(ptr_two);
        });

    ASSERT_EQ(this->m_allocator->getCurrentSize(), 62);
    EXPECT_NO_THROW(this->m_allocator->release());

    ASSERT_LE(this->m_allocator->getActualSize(), this->m_initial_pool_size);

    ASSERT_NO_THROW(
        {
            this->m_allocator->deallocate(ptr_one);
        });

    ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);

    EXPECT_NO_THROW(this->m_allocator->release());

    ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
    ASSERT_LE(this->m_allocator->getActualSize(), 0);
}

TYPED_TEST(PrimaryPoolTest, MissingBlocks)
{
    void* ptr_one{nullptr};
    void* ptr_two{nullptr};

    ASSERT_NO_THROW(
        {
            ptr_one = this->m_allocator->allocate(128);
            ptr_two = this->m_allocator->allocate(44);
            this->m_allocator->deallocate(ptr_one);
            this->m_allocator->deallocate(ptr_two);
        });

    ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
    ASSERT_GE(this->m_allocator->getActualSize(), 0);

    this->m_allocator->release();

    ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
    ASSERT_EQ(this->m_allocator->getActualSize(), 0);
}

TYPED_TEST(PrimaryPoolTest, largestavailable)
{
    using Pool  = typename TestFixture::Pool;
    const int num_allocs = 16;

    auto dynamic_pool =
                    umpire::util::unwrap_allocator<Pool>(*this->m_allocator);

    ASSERT_NE(dynamic_pool, nullptr);

    ASSERT_NO_THROW(
        {
            void* ptr{ this->m_allocator->allocate(1024) };

            this->m_allocator->deallocate(ptr);
        });

    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), this->m_initial_pool_size);

    void* ptrs[num_allocs];

    for ( int i{0}; i < num_allocs; ++i ) {
        ASSERT_NO_THROW(
            {
                ptrs[i] = this->m_allocator->allocate(1024);
            });

        ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(),
                                    ( (num_allocs-(i+1)) * 1024) );
    }

    for ( int i{0}; i < num_allocs; i += 2 ) {
        ASSERT_NO_THROW(
            {
                this->m_allocator->deallocate(ptrs[i]);
            });

        ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), 1024);
    }

    for ( int i{1}; i < num_allocs; i += 2 ) {
        const int largest_block{
                        ((i+2) < num_allocs) ? (i+2) * 1024 : (i+1) * 1024 };
        ASSERT_NO_THROW(
            {
                this->m_allocator->deallocate(ptrs[i]);
            });
        ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), largest_block);
    }
}

TYPED_TEST(PrimaryPoolTest, coalesce)
{
    using Pool  = typename TestFixture::Pool;

    auto dynamic_pool =
                    umpire::util::unwrap_allocator<Pool>(*this->m_allocator);

    ASSERT_NE(dynamic_pool, nullptr);

    void* ptr_one{nullptr};
    void* ptr_two{nullptr};

    ASSERT_NO_THROW(
        {
            ptr_one = this->m_allocator->allocate( 1 + this->m_initial_pool_size - this->m_min_pool_growth_size );
            ptr_two = this->m_allocator->allocate( this->m_min_pool_growth_size );
        });

    ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
    ASSERT_NO_THROW(
        {
            this->m_allocator->deallocate(ptr_two);
        });

    dynamic_pool->coalesce();

    ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);

    ASSERT_NO_THROW(
        {
            this->m_allocator->deallocate(ptr_one);
        });

    dynamic_pool->coalesce();

    ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);

    ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
    ASSERT_EQ(dynamic_pool->getActualSize(), this->m_initial_pool_size + this->m_alignment);
    ASSERT_EQ(this->m_allocator->getHighWatermark(),
             1 + this->m_initial_pool_size );
}
