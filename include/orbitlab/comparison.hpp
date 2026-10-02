#pragma once
#include "physics.hpp"
#include <algorithm>
namespace orbitlab {
struct Comparison { bool passed=true;double max_absolute=0,max_normalized=0; };
template<class A,class B> Comparison acceleration_matches(const Acceleration<A>& a,const Acceleration<B>& b,double atol,double rtol) {
 Comparison c;
 auto invalid=[&](){return Comparison{false,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};};
 if(!std::isfinite(atol)||!std::isfinite(rtol)||atol<=0||rtol<0||a.x.empty()) return invalid();
 std::size_t n=a.x.size();
 if(a.y.size()!=n||a.z.size()!=n||b.x.size()!=n||b.y.size()!=n||b.z.size()!=n) return invalid();
 for(std::size_t i=0;i<n;++i) {
  for(double v:{double(a.x[i]),double(a.y[i]),double(a.z[i]),double(b.x[i]),double(b.y[i]),double(b.z[i])}) if(!std::isfinite(v)) return invalid();
  double err=std::hypot(double(a.x[i])-double(b.x[i]),double(a.y[i])-double(b.y[i]),double(a.z[i])-double(b.z[i]));
  double norm=std::hypot(double(b.x[i]),double(b.y[i]),double(b.z[i]));
  double ratio=err/(atol+rtol*norm);
  if(!std::isfinite(ratio)) return invalid();
  c.max_absolute=std::max(c.max_absolute,err);c.max_normalized=std::max(c.max_normalized,ratio);c.passed=c.passed&&ratio<=1;
 }
 return c;
}
}
