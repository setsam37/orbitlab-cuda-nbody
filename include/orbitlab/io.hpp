#pragma once
#include "diagnostics.hpp"
#include <istream>
#include <ostream>
#include <iomanip>
#include <sstream>
#include <charconv>
namespace orbitlab {
inline void check_output(std::ostream& out) { if(!out) throw std::runtime_error("output stream failed"); }
template<class R> void write_state(std::ostream& out,const State<R>& s) {
 validate_state(s);out<<std::setprecision(std::numeric_limits<R>::max_digits10)<<"id,mass,x,y,z,vx,vy,vz\n";
 for(std::size_t i=0;i<s.size();++i) out<<i<<','<<s.mass[i]<<','<<s.x[i]<<','<<s.y[i]<<','<<s.z[i]<<','<<s.vx[i]<<','<<s.vy[i]<<','<<s.vz[i]<<'\n';
 check_output(out);
}
template<class R> State<R> read_state(std::istream& in) {
 std::string line;
 auto get_line=[&](){bool ok=bool(std::getline(in,line));if(!line.empty()&&line.back()=='\r') line.pop_back();return ok;};
 if(!get_line()||line!="id,mass,x,y,z,vx,vy,vz") throw std::invalid_argument("invalid initial-state CSV header");
 State<R> s;std::vector<R>* fields[]={&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz};
 while(get_line()) {
  std::stringstream row(line);std::string token;std::vector<std::string> cells;
  while(std::getline(row,token,',')) cells.push_back(token);
  if(cells.size()!=8||line.empty()||line.back()==',') throw std::invalid_argument("initial-state CSV row must have eight fields");
  std::size_t id=0;auto p=std::from_chars(cells[0].data(),cells[0].data()+cells[0].size(),id);
  if(p.ec!=std::errc()||p.ptr!=cells[0].data()+cells[0].size()||id!=s.size()) throw std::invalid_argument("CSV IDs must be consecutive from zero");
  validate_count<R>(s.size()+1);
  for(int k=0;k<7;++k) {
   try { std::size_t end=0;double v=std::stod(cells[k+1],&end);
    if(end!=cells[k+1].size()||!std::isfinite(v)||!std::isfinite(R(v))) throw std::invalid_argument("nonfinite or malformed CSV value");
    fields[k]->push_back(R(v));
   } catch(const std::exception&) { throw std::invalid_argument("nonfinite or malformed CSV value"); }
  }
 }
 if(in.bad()) throw std::runtime_error("input stream failed");
 validate_state(s);return s;
}
template<class R> void write_frame(std::ostream& out,std::size_t step,double time,const State<R>& s) {
 validate_state(s);if(!std::isfinite(time)||time<0) throw std::invalid_argument("invalid frame time");
 out<<std::setprecision(17);
 for(std::size_t i=0;i<s.size();++i) out<<step<<','<<time<<','<<i<<','<<s.x[i]<<','<<s.y[i]<<','<<s.z[i]<<','<<s.vx[i]<<','<<s.vy[i]<<','<<s.vz[i]<<'\n';
 check_output(out);
}
inline void write_diagnostic(std::ostream& out,std::size_t step,double time,const Diagnostics& d) {
 out<<std::setprecision(17)<<step<<','<<time<<','<<d.energy;
 for(double v:d.momentum) out<<','<<v;
 for(double v:d.angular_momentum) out<<','<<v;
 out<<'\n';check_output(out);
}
}
