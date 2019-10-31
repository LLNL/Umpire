//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_size_map_HPP
#define UMPIRE_size_map_HPP

#include <iostream>

#include <cmath>
#include <list>

#include <algorithm>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

namespace umpire {
namespace util {

struct size_map_iterator_begin{};
struct size_map_iterator_end{};

template<typename Key, typename Value, int Bins>
class size_map
{

  class iterator_ : 
    public std::iterator<std::bidirectional_iterator_tag, Value>
  {
    using Bin = std::size_t;
    using Index = typename std::vector<Value>::iterator;
    using Map = size_map<Key, Value, Bins>;

  public: 

    friend class size_map<Key, Value, Bins>;

    iterator_() : map{nullptr}, bin{Bins}, idx{} {}

    iterator_(Map* m, Bin b) :
      map{m}, bin{b}, idx{m->entries[bin].begin()} {}

    iterator_(Map* m, Bin b, Index i) :
      map{m}, bin{b}, idx{i} {}

    iterator_(Map* m, size_map_iterator_begin) :
      iterator(m, 0)
    {}

    iterator_(Map* m,size_map_iterator_end) :
      map{m}, bin{Bins}, idx{} {}

    Value& operator*()
    {
      return *idx; //map->entries[bin][idx];
    }

    Value* operator->(){
      return &(*idx); //&(map->entries[bin][idx]);
    }

    iterator_& operator++()
    {
      idx++;

      if (idx == map->entries[bin].end()) {
        bin++;
        if (bin < Bins)
          idx = map->entries[bin].begin();
      }

      return *this;
    }

    iterator_& operator--()
    {
      if (idx == map->entries[bin].begin()) {
        bin--;
        if (bin >= 0)
          idx = map->entries[bin].end();
      }

      idx--;
      return *this;
    }

    bool operator==(const iterator_& other) const
    {
      return (bin == other.bin) && (idx == other.idx);
    }

    bool operator!=(const iterator_& other) const
    {
      return !(*this == other);
    }

    private: 
      Map* map;
      Bin bin;
      Index idx;
  };

  friend class iterator_;

  public:

  using iterator = iterator_;

  size_map() :
    entries{}
  {
  }
    
  ~size_map() {
  }

  size_map(const size_map&) = delete;

  iterator insert(std::pair<Key, Value>&& pair)
  {
    auto key = pair.first;
    auto bin = std::size_t{std::ceil(std::log2(key))};

    auto it = (bin >= Bins) ? 
      overflow.insert(overflow.end(), pair.second)
      : entries[bin].insert(entries[bin].end(), pair.second);
    return iterator{this, bin, it};
  }


  iterator lower_bound(const Key& key)
  {
    auto bin = std::size_t{std::ceil(std::log2(key))};

    if (bin >= Bins) {
      auto entry = std::find_if(overflow.begin(), overflow.end(), [=] (Value & v) {
          return v->size >= key;
      });
      return (entry == overflow.end()) ? end() : iterator{this, Bins, entry};
    } else  {
      return (entries[bin].begin() == entries[bin].end()) ? 
        end() : iterator{this, bin};
    }
  }

  iterator erase(iterator it) {
    auto bin = it.bin;
    auto index = it.idx;
    auto prev = index;
    --prev;

    if (bin >= Bins) {
      prev = overflow.erase(index);
    } else {
      entries[bin].erase(index);
    }

    return iterator{this, bin, prev};
  }

  iterator begin() {
    return iterator{this, size_map_iterator_end{}};
  }

  iterator end(){
    return iterator{this, size_map_iterator_end{}};
  }

  std::size_t size() {
    return Bins;
  }

  private:
  std::list<Value> entries[Bins];
  std::list<Value> overflow;
};

}
}

#pragma GCC diagnostic pop
#endif
