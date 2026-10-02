#pragma once
#include "physics.hpp"
namespace orbitlab {
enum class Integrator { Verlet,Euler };
template<class R> class CpuSimulator {
 State<R> s_;ForceConfig config_;Integrator integrator_;Acceleration<R> a_;
public:
 CpuSimulator(State<R> s,ForceConfig c,Integrator method=Integrator::Verlet):s_(std::move(s)),config_(c),integrator_(method),a_(cpu_acceleration(s_,c)) {}
 void step(R dt) {
  if(!std::isfinite(dt)||dt<=0) throw std::invalid_argument("dt must be finite and positive");
  auto update=[&](std::vector<R>& pos,std::vector<R>& vel,const std::vector<R>& olda){
   for(std::size_t i=0;i<s_.size();++i) {
    pos[i]+=vel[i]*dt+(integrator_==Integrator::Verlet?R(.5)*olda[i]*dt*dt:R(0));
    if(integrator_==Integrator::Euler) vel[i]+=olda[i]*dt;
   }
  };
  update(s_.x,s_.vx,a_.x);update(s_.y,s_.vy,a_.y);update(s_.z,s_.vz,a_.z);
  auto next=cpu_acceleration(s_,config_);
  if(integrator_==Integrator::Verlet) for(std::size_t i=0;i<s_.size();++i) {
   s_.vx[i]+=R(.5)*(a_.x[i]+next.x[i])*dt;
   s_.vy[i]+=R(.5)*(a_.y[i]+next.y[i])*dt;
   s_.vz[i]+=R(.5)*(a_.z[i]+next.z[i])*dt;
  }
  a_=std::move(next);validate_state(s_);
 }
 void reset(State<R> s) { auto next=cpu_acceleration(s,config_);s_=std::move(s);a_=std::move(next); }
 const State<R>& state() const { return s_; }
};
}
