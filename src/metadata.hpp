#pragma once
#include "orbitlab/options.hpp"
#include <fstream>
namespace orbitlab {
inline void metadata(const Options& o) {
 std::ofstream f(o.output/"metadata.json");
 f<<std::setprecision(17)<<"{\n\"n\":"<<o.n<<",\"steps\":"<<o.steps<<",\"seed\":"<<o.seed<<",\"dt\":"<<o.dt<<",\"epsilon\":"<<o.epsilon
  <<",\"block_size\":"<<o.block<<",\"sample_every\":"<<o.sample_every<<",\"precision\":\""<<o.precision<<"\",\"backend\":\""<<o.backend<<"\",\"integrator\":\""<<o.integrator<<"\",\"mode\":\""<<o.mode<<"\",\"units\":\"dimensionless, G=1\"\n}\n";
 f.flush();check_output(f);
}
}
