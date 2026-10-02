#pragma once
#include "physics.hpp"
namespace orbitlab {
enum class Integrator { Verlet,Euler };
template<class R> class CpuSimulator {
 State<R> s_;ForceConfig config_;Integrator integrator_;Acceleration<R> a_,next_;volatile R force_sink_=0;
public:
 CpuSimulator(State<R> s,ForceConfig c,Integrator method=Integrator::Verlet):s_(std::move(s)),config_(c),integrator_(method),a_(cpu_acceleration(s_,c)),next_(a_) {}
 void step(R dt) {
  if(!std::isfinite(dt)||dt<=0) throw std::invalid_argument("dt must be finite and positive");
  auto update=[&](std::vector<R>& pos,std::vector<R>& vel,const std::vector<R>& olda){
   for(std::size_t i=0;i<s_.size();++i) {
    pos[i]+=vel[i]*dt+(integrator_==Integrator::Verlet?R(.5)*olda[i]*dt*dt:R(0));
    if(integrator_==Integrator::Euler) vel[i]+=olda[i]*dt;
   }
  };
  update(s_.x,s_.vx,a_.x);update(s_.y,s_.vy,a_.y);update(s_.z,s_.vz,a_.z);
  cpu_acceleration_into(s_,config_,next_);
  if(integrator_==Integrator::Verlet) for(std::size_t i=0;i<s_.size();++i) {
   s_.vx[i]+=R(.5)*(a_.x[i]+next_.x[i])*dt;
   s_.vy[i]+=R(.5)*(a_.y[i]+next_.y[i])*dt;
   s_.vz[i]+=R(.5)*(a_.z[i]+next_.z[i])*dt;
  }
  std::swap(a_,next_);validate_state(s_);
 }
 void reset(const State<R>& s) { cpu_acceleration_into(s,config_,next_);s_=s;std::swap(a_,next_); }
 const State<R>& state() const { return s_; }
 void force_only() { cpu_acceleration_into(s_,config_,a_);force_sink_+=a_.x[0]; }
 Acceleration<R> acceleration_snapshot() const { return a_; }
};
}
