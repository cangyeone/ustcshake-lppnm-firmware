#pragma once
// Portable C++17 W8A8 incremental LPPNM. No Torch, ONNX Runtime, or ESP-DL.
// Model buffers allocate once during construction; push/reset do not allocate.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lppnm {
constexpr int kWindow = 3072, kHop = 128, kClasses = 7, kMaxOutput = 32;
enum Kind { Input, Conv, Up, Add, Cat };
struct NodeSpec {
  Kind kind;
  int parent0, parent1, channels, kernel, stride, padding, groups;
  bool relu;
  int capacity;
  float scale;
  const int8_t* weight;
  const int32_t* bias;
  const int64_t* multiplier;
};
struct RawOutput {
  int64_t bin;
  std::array<int8_t, kClasses> logits;
  int8_t offset;
};
struct Prediction {
  int64_t bin;
  double sample;
  std::array<float, kClasses> probability;
};

inline int8_t requant(int64_t v) {
  int64_t q = v >= 0 ? (v + (1LL << 23)) >> 24 : -((-v + (1LL << 23)) >> 24);
  return static_cast<int8_t>(std::clamp<int64_t>(q, -127, 127));
}

class Ring {
 public:
  int channels = 0, capacity = 0;
  int64_t base = 0, end = 0;
  std::vector<int8_t> data;
  Ring(int c, int cap) : channels(c), capacity(cap), data(c*cap) {}
  void reset() { base = end = 0; }
  int8_t get(int64_t t, int c) const {
    if (t < 0) return 0; // Only the true stream origin has left zero padding.
    assert(t >= base && t < end && c >= 0 && c < channels);
    return data[(t % capacity)*channels+c];
  }
  const int8_t* frame(int64_t t) const {
    if (t<0) return nullptr;
    assert(t>=base && t<end);
    return data.data()+(t%capacity)*channels;
  }
  int8_t* write_frame() {
    assert(end-base<capacity);
    return data.data()+(end%capacity)*channels;
  }
};

class Engine {
 public:
  Engine(const NodeSpec* specs, int count, int class_head, int time_head)
      : specs_(specs), count_(count), cl_(class_head), tm_(time_head), next_(count), needed_(count) {
    buffers_.reserve(count);
    for (int i=0;i<count;i++) buffers_.emplace_back(specs[i].channels,specs[i].capacity);
  }
  void reset() {
    for (auto& b : buffers_) b.reset();
    std::fill(next_.begin(),next_.end(),0);
    emitted_ = 0; macs_ = 0;
  }
  uint64_t macs() const { return macs_; }
  uint64_t samples_seen() const { return buffers_[0].end; }
  size_t cache_bytes() const {
    size_t total=0;
    for (const auto& b : buffers_) total += b.data.size();
    return total;
  }
  size_t allocated_bytes() const {
    return sizeof(*this)+cache_bytes()+buffers_.capacity()*sizeof(Ring)
      +(next_.capacity()+needed_.capacity())*sizeof(int64_t);
  }
  Prediction decode(const RawOutput& r) const {
    Prediction p{};
    p.bin=r.bin;
    float maximum=-INFINITY,sum=0;
    for (int c=0;c<kClasses;c++) maximum=std::max(maximum,r.logits[c]*specs_[cl_].scale);
    for (int c=0;c<kClasses;c++) {
      p.probability[c]=std::exp(r.logits[c]*specs_[cl_].scale-maximum);
      sum+=p.probability[c];
    }
    for (auto& v:p.probability) v/=sum;
    p.sample=static_cast<double>(r.bin)*8.0+8.0/(1.0+std::exp(-r.offset*specs_[tm_].scale));
    return p;
  }
  bool push_normalized(const float* x, RawOutput* output, int& output_count) {
    output_count=0;
    // Validate first: rejected data must never partially mutate stream state.
    for (int j=0;j<kHop*3;j++) if (!std::isfinite(x[j])) return false;
    Ring& input=buffers_[0];
    for (int t=0;t<kHop;t++) {
      auto* row=input.write_frame();
      for (int c=0;c<3;c++) {
        // Input scale is exactly 1/127 in Python; avoid FP32 scale reciprocal drift.
        float v=std::clamp(x[t*3+c],-1.0f,1.0f);
        int q=static_cast<int>(std::round(static_cast<double>(v)*127.0));
        row[c]=static_cast<int8_t>(q);
      }
      input.end++;
    }
    for (int i=1;i<count_;i++) {
      const auto& n=specs_[i];
      auto& dest=buffers_[i];
      const auto& a=buffers_[n.parent0];
      int64_t stop=0;
      if (n.kind==Conv) stop=a.end>n.padding ? (a.end-1-n.padding)/n.stride+1 : 0;
      else if (n.kind==Up) stop=a.end*2;
      else stop=std::min(a.end,buffers_[n.parent1].end);
      for (int64_t t=next_[i];t<stop;t++) {
        auto* row=dest.write_frame();
        if (n.kind==Conv) {
          int in_per=a.channels/n.groups, out_per=n.channels/n.groups;
          // Resolve ring addresses once per kernel tap, not once per MAC.
          // In particular avoid a software 64-bit modulo inside the C3 MAC loop.
          assert(n.kernel<=5);
          const int8_t* frames[5]{};
          for (int k=0;k<n.kernel;k++) frames[k]=a.frame(t*n.stride-n.padding+k);
          for (int oc=0;oc<n.channels;oc++) {
            int32_t acc=n.bias[oc];
            int first=(oc/out_per)*in_per;
            for (int ic=0;ic<in_per;ic++) for (int k=0;k<n.kernel;k++) {
              int v=frames[k]?frames[k][first+ic]:0;
              acc+=v*n.weight[(oc*in_per+ic)*n.kernel+k];
            }
            int8_t value=requant(static_cast<int64_t>(acc)*n.multiplier[oc]);
            if (n.relu && value<0) value=0;
            row[oc]=value;
          }
          macs_+=static_cast<uint64_t>(n.channels)*in_per*n.kernel;
        } else if (n.kind==Up) {
          auto* source=a.frame(t/2);
          for (int c=0;c<n.channels;c++) row[c]=source[c];
        } else if (n.kind==Add) {
          const auto& b=buffers_[n.parent1];
          auto* pa=a.frame(t);auto* pb=b.frame(t);
          for (int c=0;c<n.channels;c++) row[c]=requant(
            static_cast<int64_t>(pa[c])*n.multiplier[0]+
            static_cast<int64_t>(pb[c])*n.multiplier[1]);
        } else if (n.kind==Cat) {
          const auto& b=buffers_[n.parent1];
          auto* pa=a.frame(t);auto* pb=b.frame(t);
          for (int c=0;c<a.channels;c++) row[c]=requant(static_cast<int64_t>(pa[c])*n.multiplier[0]);
          for (int c=0;c<b.channels;c++) row[a.channels+c]=requant(static_cast<int64_t>(pb[c])*n.multiplier[1]);
        }
        dest.end++;
      }
      next_[i]=stop;
    }
    int64_t stop=std::min(buffers_[cl_].end,buffers_[tm_].end);
    assert(stop-emitted_ <= kMaxOutput);
    while (emitted_<stop) {
      auto& r=output[output_count++];r.bin=emitted_;
      for (int c=0;c<kClasses;c++) r.logits[c]=buffers_[cl_].get(emitted_,c);
      r.offset=buffers_[tm_].get(emitted_,0);
      emitted_++;
    }
    trim();
    return true;
  }
 private:
  void trim() {
    for (int i=0;i<count_;i++) needed_[i]=buffers_[i].end;
    for (int i=1;i<count_;i++) {
      const auto& n=specs_[i];
      int64_t start=next_[i];
      if (n.kind==Conv) start=std::max<int64_t>(0,start*n.stride-n.padding);
      if (n.kind==Up) start/=2;
      needed_[n.parent0]=std::min(needed_[n.parent0],start);
      if (n.parent1>=0) needed_[n.parent1]=std::min(needed_[n.parent1],start);
    }
    needed_[cl_]=std::min(needed_[cl_],emitted_);
    needed_[tm_]=std::min(needed_[tm_],emitted_);
    for (int i=0;i<count_;i++) buffers_[i].base=needed_[i];
  }
  const NodeSpec* specs_;
  int count_,cl_,tm_;
  std::vector<Ring> buffers_;
  std::vector<int64_t> next_,needed_;
  int64_t emitted_=0;
  uint64_t macs_=0;
};

class Normalizer {
 public:
  // Monotonic queues store uint16 ring slots, not full timestamps/float copies.
  std::array<float,kWindow> history{};
  std::array<uint16_t,kWindow> low{},high{};
  int low_head=0,high_head=0,low_count=0,high_count=0;
  uint64_t count=0;
  double total=0;
  void reset() { low_head=high_head=low_count=high_count=0;count=0;total=0; }
  float process(float v) {
    int slot=count%kWindow;
    if (count>=kWindow) {
      if (low_count && low[low_head]==slot) { low_head=(low_head+1)%kWindow;low_count--; }
      if (high_count && high[high_head]==slot) { high_head=(high_head+1)%kWindow;high_count--; }
      total-=history[slot];
    }
    history[slot]=v;total+=v;
    while (low_count && history[low[(low_head+low_count-1)%kWindow]]>=v) low_count--;
    while (high_count && history[high[(high_head+high_count-1)%kWindow]]<=v) high_count--;
    low[(low_head+low_count++)%kWindow]=slot;
    high[(high_head+high_count++)%kWindow]=slot;
    count++;
    double mean=total/std::min<uint64_t>(count,kWindow);
    double scale=std::max(std::abs(history[low[low_head]]-mean),
                          std::abs(history[high[high_head]]-mean))+1e-6;
    return static_cast<float>((v-mean)/scale);
  }
};

class OnlinePicker {
 public:
  Engine engine;
  Normalizer normalizer;
  std::array<float,kHop*3> normalized{};
  OnlinePicker(const NodeSpec* specs,int count,int cl,int tm) : engine(specs,count,cl,tm) {}
  void reset() { engine.reset();normalizer.reset(); }
  size_t allocated_bytes() const { return sizeof(*this)-sizeof(Engine)+engine.allocated_bytes(); }
  bool push(const float* raw,RawOutput* output,int& n) {
    n=0;
    for (int i=0;i<kHop;i++) if (!std::isfinite(raw[i])) return false;
    for (int i=0;i<kHop;i++) {
      float v=normalizer.process(raw[i]);
      normalized[i*3]=normalized[i*3+1]=normalized[i*3+2]=v;
    }
    if (!engine.push_normalized(normalized.data(),output,n)) return false;
    if (normalizer.count<kWindow) n=0;
    return true;
  }
};
} // namespace lppnm
