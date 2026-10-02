#pragma once
#include "state.hpp"
#include <random>
namespace orbitlab {
inline void validate_epsilon(double e) {
 if(!std::isfinite(e)||e<=0) throw std::invalid_argument("epsilon must be finite and positive");
}
template<class R> State<R> circular_pair(double e) {
 validate_epsilon(e);
 R v=static_cast<R>(std::sqrt(.5/std::pow(1+e*e,1.5)));
 State<R> s{{1,1},{R(-.5),R(.5)},{0,0},{0,0},{0,0},{-v,v},{0,0}};
 validate_state(s);return s;
}
template<class R> State<R> cloud(std::size_t n, std::uint64_t seed) {
 validate_count<R>(n);State<R> s;
 for(auto* field:{&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz}) field->resize(n);
 std::mt19937_64 rng(seed);
 // Define conversion explicitly; distribution implementations need not match.
 auto uniform=[&](){return static_cast<double>(rng()>>11)*0x1.0p-53;};
 for(std::size_t i=0;i<n;++i) {
  s.mass[i]=R(.5+uniform());
  s.x[i]=R(2*uniform()-1);s.y[i]=R(2*uniform()-1);s.z[i]=R(2*uniform()-1);
  s.vx[i]=R(.2*uniform()-.1);s.vy[i]=R(.2*uniform()-.1);s.vz[i]=R(.2*uniform()-.1);
 }
 validate_state(s);return s;
}
inline double circular_period(double e) {
 validate_epsilon(e);return 2*std::acos(-1.)/(2*std::sqrt(.5/std::pow(1+e*e,1.5)));
}
}
