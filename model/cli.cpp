#include "runtime.hpp"
#include "artifacts/model.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc,char** argv) {
  if (argc!=4 || (std::string(argv[1])!="--raw" && std::string(argv[1])!="--normalized")) {
    std::cerr<<"Usage: stream_cli --raw|--normalized input_float32.bin output.csv\n";return 2;
  }
  bool normalized=std::string(argv[1])=="--normalized";
  std::ifstream input(argv[2],std::ios::binary);
  std::ofstream out(argv[3]);
  if (!input || !out) return 2;
  lppnm::OnlinePicker picker(lppnm_model::nodes,lppnm_model::count,lppnm_model::class_head,lppnm_model::time_head);
  std::array<float,lppnm::kHop*3> block{};
  std::array<lppnm::RawOutput,lppnm::kMaxOutput> rows{};
  size_t bytes=lppnm::kHop*(normalized?3:1)*sizeof(float);
  double total_ms=0,maximum_ms=0;
  uint64_t blocks=0;
  out<<"bin,c0,c1,c2,c3,c4,c5,c6,offset,emitted_after_samples\n";
  while (input.read(reinterpret_cast<char*>(block.data()),bytes)) {
    int n=0;
    auto t=std::chrono::steady_clock::now();
    bool ok=normalized?picker.engine.push_normalized(block.data(),rows.data(),n):picker.push(block.data(),rows.data(),n);
    if (!ok) { std::cerr<<"Nonfinite input; reset before resuming a discontinuous stream.\n";return 3; }
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
    total_ms+=ms;maximum_ms=std::max(maximum_ms,ms);blocks++;
    for (int i=0;i<n;i++) {
      out<<rows[i].bin;
      for (auto q:rows[i].logits) out<<','<<static_cast<int>(q);
      out<<','<<static_cast<int>(rows[i].offset)<<','<<picker.engine.samples_seen()<<'\n';
    }
  }
  // Never pad an incomplete final block as though real samples had arrived.
  if (input.gcount()!=0) { std::cerr<<"Incomplete 128-sample block; buffer it until more input arrives.\n";return 4; }
  std::cerr<<"{\"blocks\":"<<blocks<<",\"mean_ms_host\":"<<(blocks?total_ms/blocks:0)
           <<",\"max_ms_host\":"<<maximum_ms<<",\"allocated_bytes_host_abi\":"<<picker.allocated_bytes()
           <<",\"macs\":"<<picker.engine.macs()<<"}\n";
  return 0;
}
