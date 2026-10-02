#pragma once
#include "orbitlab/comparison.hpp"
#include <chrono>
namespace orbitlab {
template<class R> void snapshot_into(CpuSimulator<R>& sim,State<R>& s) { s=sim.state(); }
#ifdef ORBITLAB_HAS_CUDA
template<class R> void snapshot_into(GpuSimulator<R>& sim,State<R>& s) { sim.download_into(s); }
#endif
template<class R> double measure(CpuSimulator<R>& sim,const Options& o,int count) {
 auto begin=std::chrono::steady_clock::now();
 for(int k=0;k<count;++k) { if(o.mode=="force") sim.force_only();else sim.step(R(o.dt)); }
 return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/count;
}
#ifdef ORBITLAB_HAS_CUDA
template<class R> double measure(GpuSimulator<R>& sim,const Options& o,int count) { return sim.timed_batch(o.mode=="force",count,R(o.dt)); }
#endif
template<class R,class Sim> void benchmark(Sim& sim,const State<R>& initial,const Options& o) {
 // Same-precision baselines; accuracy reference starts from identical stored values.
 State<double> reference;const std::vector<R>* src[]={&initial.mass,&initial.x,&initial.y,&initial.z,&initial.vx,&initial.vy,&initial.vz};
 std::vector<double>* dst[]={&reference.mass,&reference.x,&reference.y,&reference.z,&reference.vx,&reference.vy,&reference.vz};
 for(int k=0;k<7;++k) dst[k]->assign(src[k]->begin(),src[k]->end());
 auto expected=cpu_acceleration(reference,{o.epsilon});auto actual=sim.acceleration_snapshot();
 auto comparison=acceleration_matches(actual,expected,sizeof(R)==4?1e-5:1e-11,sizeof(R)==4?3e-4:1e-10);
 if(!comparison.passed) throw std::runtime_error("benchmark fixture failed force agreement");
 std::filesystem::create_directories(o.output);std::ofstream csv(o.output/"batches.csv");
 csv<<std::setprecision(17)<<"batch,per_call_ms,repetitions,warmups,n,backend,precision,block_size,mode,max_acceleration_error\n";
 int warmups=o.backend=="cpu"?1:5,count=o.backend=="cpu"?5:20;
 if(o.mode=="end-to-end") { count=20;warmups=1; }
 State<R> final_state=initial;
 auto wall_run=[&](){
  auto begin=std::chrono::steady_clock::now();sim.reset(initial);
  for(int k=0;k<20;++k) sim.step(R(o.dt));snapshot_into(sim,final_state);validate_state(final_state);
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/20;
 };
 if(o.mode=="end-to-end") wall_run();
 else { for(int k=0;k<warmups;++k) { if(o.mode=="force") sim.force_only();else sim.step(R(o.dt)); } auto check=snapshot(sim);validate_state(check); }
 for(int batch=0;batch<5;++batch) {
  double ms;
  if(o.mode=="end-to-end") ms=wall_run();
  else { sim.reset(initial);ms=measure(sim,o,count);auto check=snapshot(sim);validate_state(check); }
  if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("invalid benchmark timing");
  csv<<batch<<','<<ms<<','<<count<<','<<warmups<<','<<o.n<<','<<o.backend<<','<<o.precision<<','<<o.block<<','<<o.mode<<','<<comparison.max_absolute<<'\n';
 }
 csv.flush();check_output(csv);metadata(o);std::ofstream states(o.output/"initial-state.csv");write_state(states,initial);states.flush();check_output(states);
 std::cout<<"BENCHMARK_OK backend="<<o.backend<<" precision="<<o.precision<<" N="<<o.n<<" mode="<<o.mode<<" max_acceleration_error="<<comparison.max_absolute<<"\n";
}
}
