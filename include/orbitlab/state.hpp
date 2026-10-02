#pragma once
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstdint>
namespace orbitlab {
template<class R> struct State {
  std::vector<R> mass,x,y,z,vx,vy,vz;
  std::size_t size() const { return mass.size(); }
};
template<class R> void validate_count(std::size_t n) {
  // Bound byte arithmetic and CUDA's integer indexing before allocating.
  if(n==0 || n>static_cast<std::size_t>(std::numeric_limits<int>::max()-256) ||
     n>std::numeric_limits<std::size_t>::max()/(13*sizeof(R)))
    throw std::invalid_argument("particle count is zero or too large");
}
template<class R> void validate_state(const State<R>& s) {
  validate_count<R>(s.size());
  for(const auto* field:{&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz}) {
    if(field->size()!=s.size()) throw std::invalid_argument("state component lengths differ");
    for(R value:*field) if(!std::isfinite(value)) throw std::invalid_argument("state contains NaN/Inf");
  }
  for(R m:s.mass) if(m<=0) throw std::invalid_argument("mass must be positive");
}
}
