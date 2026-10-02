#pragma once
#include "fixtures.hpp"
namespace orbitlab {
struct ForceConfig { double epsilon=.01; };
template<class R> struct Acceleration { std::vector<R> x,y,z; };
template<class R> R softening_squared(ForceConfig c) {
 validate_epsilon(c.epsilon);R e=R(c.epsilon),e2=e*e;
 if(!std::isfinite(e2)||e2<=0) throw std::invalid_argument("epsilon squared is not representable in selected precision");
 return e2;
}
template<class R> Acceleration<R> cpu_acceleration(const State<R>& s,ForceConfig c) {
 validate_state(s);R e2=softening_squared<R>(c);
 Acceleration<R> a{std::vector<R>(s.size()),std::vector<R>(s.size()),std::vector<R>(s.size())};
 for(std::size_t i=0;i<s.size();++i) {
  R ax=0,ay=0,az=0;
  for(std::size_t j=0;j<s.size();++j) {
   if(i==j) continue;
   R dx=s.x[j]-s.x[i],dy=s.y[j]-s.y[i],dz=s.z[j]-s.z[i];
   R r2=dx*dx+dy*dy+dz*dz+e2;
   R scale=s.mass[j]/(r2*std::sqrt(r2));
   ax+=dx*scale;ay+=dy*scale;az+=dz*scale;
  }
  if(!std::isfinite(ax)||!std::isfinite(ay)||!std::isfinite(az)) throw std::runtime_error("nonfinite acceleration");
  a.x[i]=ax;a.y[i]=ay;a.z[i]=az;
 }
 return a;
}
}
