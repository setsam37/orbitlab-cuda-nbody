#pragma once
#include "physics.hpp"
#include <array>
namespace orbitlab {
struct Diagnostics { double energy=0;std::array<double,3> momentum{},angular_momentum{}; };
template<class R> Diagnostics diagnostics(const State<R>& s,ForceConfig c) {
 validate_state(s);double e2=softening_squared<double>(c);Diagnostics d;
 for(std::size_t i=0;i<s.size();++i) {
  double m=s.mass[i],x=s.x[i],y=s.y[i],z=s.z[i],vx=s.vx[i],vy=s.vy[i],vz=s.vz[i];
  d.energy+=.5*m*(vx*vx+vy*vy+vz*vz);
  d.momentum[0]+=m*vx;d.momentum[1]+=m*vy;d.momentum[2]+=m*vz;
  d.angular_momentum[0]+=m*(y*vz-z*vy);d.angular_momentum[1]+=m*(z*vx-x*vz);d.angular_momentum[2]+=m*(x*vy-y*vx);
  for(std::size_t j=i+1;j<s.size();++j) {
   double dx=double(s.x[j])-x,dy=double(s.y[j])-y,dz=double(s.z[j])-z;
   d.energy-=m*double(s.mass[j])/std::sqrt(dx*dx+dy*dy+dz*dz+e2);
  }
 }
 for(double v:{d.energy,d.momentum[0],d.momentum[1],d.momentum[2],d.angular_momentum[0],d.angular_momentum[1],d.angular_momentum[2]})
  if(!std::isfinite(v)) throw std::runtime_error("nonfinite diagnostic");
 return d;
}
}
