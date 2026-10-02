#include <iostream>
#include "orbitlab/options.hpp"
#include "orbitlab/io.hpp"
#include "orbitlab/cpu_simulator.hpp"
#include "metadata.hpp"
#ifdef ORBITLAB_HAS_CUDA
#include "orbitlab/cuda_simulator.hpp"
#endif
using namespace orbitlab;
template<class R> State<R> initial_state(Options& o) {
 State<R> s;
 if(!o.initial_file.empty()) { std::ifstream in(o.initial_file);if(!in) throw std::runtime_error("cannot open initial state");s=read_state<R>(in); }
 else s=o.fixture=="circular"?circular_pair<R>(o.epsilon):cloud<R>(o.n,o.seed);
 o.n=s.size();return s;
}
template<class R> State<R> snapshot(CpuSimulator<R>& sim) { return sim.state(); }
#ifdef ORBITLAB_HAS_CUDA
template<class R> State<R> snapshot(GpuSimulator<R>& sim) { return sim.download(); }
#endif
#include "benchmark.hpp"
template<class R,class Sim> void simulate(Sim& sim,const State<R>& initial,const Options& o) {
 std::filesystem::create_directories(o.output);
 std::ofstream statefile(o.output/"initial-state.csv"),trajectory(o.output/"trajectory.csv"),diag(o.output/"diagnostics.csv");
 write_state(statefile,initial);trajectory<<"step,time,id,x,y,z,vx,vy,vz\n";diag<<"step,time,energy,px,py,pz,lx,ly,lz\n";
 auto sample=[&](std::size_t k) { auto s=snapshot(sim);double time=double(k)*o.dt;write_frame(trajectory,k,time,s);write_diagnostic(diag,k,time,diagnostics(s,{o.epsilon})); };
 sample(0);
 for(std::size_t k=1;k<=o.steps;++k) { sim.step(R(o.dt));if(k%o.sample_every==0||k==o.steps) sample(k); }
 statefile.flush();trajectory.flush();diag.flush();check_output(statefile);check_output(trajectory);check_output(diag);metadata(o);
 std::cout<<"SIMULATION_OK backend="<<o.backend<<" precision="<<o.precision<<" N="<<o.n<<" steps="<<o.steps<<" output="<<o.output<<"\n";
}
template<class R> void execute(Options o) {
 if(!std::isfinite(R(o.dt))||R(o.dt)<=0) throw std::invalid_argument("dt is not representable in selected precision");
 softening_squared<R>({o.epsilon});auto s=initial_state<R>(o);
 if(o.command=="benchmark"&&o.integrator!="verlet") throw std::invalid_argument("benchmarks use Verlet");
 if(o.backend=="cpu") { CpuSimulator<R> sim(s,{o.epsilon},o.integrator=="verlet"?Integrator::Verlet:Integrator::Euler);if(o.command=="benchmark") benchmark(sim,s,o);else simulate(sim,s,o); }
 else {
#ifdef ORBITLAB_HAS_CUDA
  GpuSimulator<R> sim(s,{o.epsilon},o.backend=="cuda-basic"?GpuKernel::Basic:GpuKernel::Tiled,o.block);if(o.command=="benchmark") benchmark(sim,s,o);else simulate(sim,s,o);
#else
  throw std::runtime_error("CUDA backend unavailable: configure with -DORBITLAB_ENABLE_CUDA=ON");
#endif
 }
}
int main(int argc,char** argv) {
 try { if(argc>1&&std::string(argv[1])=="--help") { std::cout<<"OrbitLab: simulate or benchmark; see README for options\n";return 0; }
  auto o=parse_options(argc,argv);if(o.precision=="float") execute<float>(o);else execute<double>(o);return 0;
 } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n";return 1; }
}
