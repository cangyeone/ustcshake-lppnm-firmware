#include "runtime.hpp"
#include "artifacts/model.hpp"
#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>

static std::atomic<size_t> allocations{0};
void* operator new(std::size_t size) {
  allocations++;
  if (auto p=std::malloc(size?size:1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }

int main() {
  lppnm::OnlinePicker a(lppnm_model::nodes,lppnm_model::count,lppnm_model::class_head,lppnm_model::time_head);
  lppnm::OnlinePicker b(lppnm_model::nodes,lppnm_model::count,lppnm_model::class_head,lppnm_model::time_head);
  std::array<float,128> x{};
  std::array<lppnm::RawOutput,32> oa{},ob{};
  const auto before=allocations.load();
  // Exercise complete reset/replay twice. No allocations after construction.
  for (int replay=0;replay<2;replay++) {
    a.reset();b.reset();
    for (int block=0;block<80;block++) {
      for (int i=0;i<128;i++) x[i]=std::sin((block*128+i)*0.011f);
      int na=0,nb=0;
      assert(a.push(x.data(),oa.data(),na));
      assert(b.push(x.data(),ob.data(),nb));
      assert(na==nb);
      if (block<23) assert(na==0);
      for (int j=0;j<na;j++) {
        assert(oa[j].bin==ob[j].bin && oa[j].logits==ob[j].logits && oa[j].offset==ob[j].offset);
        auto p=a.engine.decode(oa[j]);
        float sum=0;
        for (auto v:p.probability) { assert(std::isfinite(v) && v>=0 && v<=1);sum+=v; }
        assert(std::abs(sum-1.0f)<1e-5f);
        assert(p.sample>=p.bin*8 && p.sample<=p.bin*8+8);
      }
      // Rejected block leaves both normalization and CNN state intact.
      x[64]=std::numeric_limits<float>::quiet_NaN();
      auto count=a.normalizer.count;auto macs=a.engine.macs();
      assert(!a.push(x.data(),oa.data(),na));
      assert(a.normalizer.count==count && a.engine.macs()==macs);
    }
  }
  assert(allocations.load()==before);
}
