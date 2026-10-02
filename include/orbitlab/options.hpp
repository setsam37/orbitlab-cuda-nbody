#pragma once
#include "fixtures.hpp"
#include <filesystem>
#include <charconv>
#include <map>
namespace orbitlab {
struct Options {
 std::string command="simulate",backend="cpu",precision="float",integrator="verlet",fixture="circular",mode="force",initial_file;
 std::filesystem::path output="results/run";
 std::size_t n=2,steps=1000,sample_every=10;std::uint64_t seed=37;int block=128;double epsilon=.01,dt=.001;
};
inline std::uint64_t integer(const std::string& value) {
 std::uint64_t n=0;auto r=std::from_chars(value.data(),value.data()+value.size(),n);
 if(r.ec!=std::errc()||r.ptr!=value.data()+value.size()) throw std::invalid_argument("invalid unsigned integer: "+value);return n;
}
inline double number(const std::string& value) {
 std::size_t end=0;double v=std::stod(value,&end);if(end!=value.size()||!std::isfinite(v)) throw std::invalid_argument("invalid finite number: "+value);return v;
}
inline Options parse_options(int argc,char** argv) {
 Options o;if(argc>1) o.command=argv[1];
 if(o.command!="simulate"&&o.command!="benchmark") throw std::invalid_argument("command must be simulate or benchmark");
 if(o.command=="benchmark") { o.fixture="cloud";o.n=512; }
 for(int i=2;i<argc;i+=2) {
  if(i+1>=argc) throw std::invalid_argument("option requires a value");std::string key=argv[i],v=argv[i+1];
  if(key=="--backend") o.backend=v;else if(key=="--precision") o.precision=v;else if(key=="--integrator") o.integrator=v;
  else if(key=="--fixture") o.fixture=v;else if(key=="--mode") o.mode=v;else if(key=="--initial-state") o.initial_file=v;else if(key=="--output") o.output=v;
  else if(key=="--seed") o.seed=integer(v);else if(key=="--n") { auto n=integer(v);validate_count<double>(n);o.n=std::size_t(n); }
  else if(key=="--steps") { auto n=integer(v);if(n>100000000) throw std::invalid_argument("steps exceeds run safety limit 100000000");o.steps=std::size_t(n); }
  else if(key=="--sample-every") { auto n=integer(v);if(n==0||n>100000000) throw std::invalid_argument("sample interval out of range");o.sample_every=std::size_t(n); }
  else if(key=="--block-size") { auto n=integer(v);if(n!=64&&n!=128&&n!=256) throw std::invalid_argument("block size must be 64, 128, or 256");o.block=int(n); }
  else if(key=="--epsilon") o.epsilon=number(v);else if(key=="--dt") o.dt=number(v);else throw std::invalid_argument("unknown option: "+key);
 }
 if(o.backend!="cpu"&&o.backend!="cuda-basic"&&o.backend!="cuda-tiled") throw std::invalid_argument("invalid backend");
 if(o.precision!="float"&&o.precision!="double") throw std::invalid_argument("invalid precision");
 if(o.integrator!="verlet"&&o.integrator!="euler") throw std::invalid_argument("invalid integrator");
 if(o.backend!="cpu"&&o.integrator!="verlet") throw std::invalid_argument("Euler is available on CPU only");
 if(o.fixture!="circular"&&o.fixture!="cloud") throw std::invalid_argument("invalid fixture");
 if(o.initial_file.empty()&&o.fixture=="circular"&&o.n!=2) throw std::invalid_argument("circular fixture requires N=2");
 if(o.mode!="force"&&o.mode!="step"&&o.mode!="end-to-end") throw std::invalid_argument("invalid benchmark mode");
 validate_epsilon(o.epsilon);
 if(o.dt<=0||!std::isfinite(o.dt*double(o.steps))) throw std::invalid_argument("invalid time step or final time");
 return o;
}
}
