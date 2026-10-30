#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include "resampler_taps.hpp"
namespace shake {
constexpr int sample_hz=250;
// Original fixed-range values, in microvolts. Index zero selects automatic range.
inline constexpr float ranges[]={50,100,200,500,1000,2000,5000,10000,20000,50000,100000};
struct Settings {
  float highpass=.5f, lowpass=10.f;
  float sensitivity=20.9454536f; // Loaded sensitivity reported by original v1.1.0.
  float threshold=.01f;
  uint16_t duration=10;
  uint8_t range=0;
  bool velocity=false, logo=true;
  bool valid() const {
    return std::isfinite(highpass)&&std::isfinite(lowpass)&&highpass>=.1f &&
      highpass<lowpass && lowpass<=100 && std::isfinite(sensitivity)&&sensitivity>0 &&
      std::isfinite(threshold)&&threshold>=.01f&&threshold<=.99f&&
      duration>=5&&duration<=60&&range<=11;
  }
};
struct Biquad {
  float b0=1,b1=0,b2=0,a1=0,a2=0,z1=0,z2=0;
  void configure(float hz,bool high) {
    float w=static_cast<float>(6.283185307179586*hz/250.0);
    float c=std::cos(w),alpha=std::sin(w)/1.41421356237f,a0=1+alpha;
    b0=(high?1+c:1-c)*.5f/a0;
    b1=(high?-(1+c):1-c)/a0;b2=b0;
    a1=-2*c/a0;a2=(1-alpha)/a0;z1=z2=0;
  }
  float push(float x) {
    float y=b0*x+z1;z1=b1*x-a1*y+z2;z2=b2*x-a2*y;return y;
  }
};
struct Bandpass {
  Biquad hp,lp; float baseline=0;
  void configure(const Settings& s,float initial=0) {
    hp.configure(s.highpass,true);lp.configure(s.lowpass,false);baseline=initial;
  }
  float push(float x) { return lp.push(hp.push(x-baseline)); }
};
// Causal 2/5 polyphase FIR. Timestamps refer to the centre of the FIR, not
// to the time its output becomes available. No zero-stuffed array is allocated.
class Resampler {
  std::array<float,81> history{};
  uint64_t input_=0, next_up_=0;
 public:
  void reset() { input_=next_up_=0;history.fill(0); }
  bool push(float x,float& output,int64_t& relative_us) {
    const uint64_t index=input_++;
    history[index%history.size()]=x;
    if (next_up_>2*index) return false;
    const uint64_t at=next_up_; next_up_+=5;
    if (at<160) return false; // Full FIR support, no fabricated startup samples.
    float sum=0;
    for (int k=at%2;k<161;k+=2) {
      const uint64_t source=(at-k)/2;
      sum+=history[source%history.size()]*resample_taps[k];
    }
    output=sum;relative_us=static_cast<int64_t>(at)*2000-160000;
    return true;
  }
};
struct Pick { int64_t time_us=0; float score=0; uint8_t phase=0; };
// A peak is confirmed after five lower-probability bins (0.4 s). Suppress
// duplicates of the same phase within one second; P and S remain independent.
class PeakDetector {
  struct State { Pick best{};int quiet=0;int64_t last=INT64_MIN/2;bool active=false; };
  std::array<State,2> state{};
 public:
  void reset() { state={}; }
  template<class Emit> void push(int64_t t,const std::array<float,7>& p,float threshold,Emit emit) {
    for (int i=0;i<2;i++) {
      auto& s=state[i];float v=p[i+1];
      if (v>=threshold && (!s.active || v>s.best.score)) {
        s.best={t,v,static_cast<uint8_t>(i+1)};s.active=true;s.quiet=0;
      } else if (s.active && ++s.quiet>=5) {
        if (s.best.time_us-s.last>=1'000'000) { emit(s.best);s.last=s.best.time_us; }
        s.active=false;s.quiet=0;
      }
    }
  }
};
enum class KeyEvent {None,Short,Long};
class Button {
  bool raw_=false,down_=false,long_=false;
  uint32_t change_=0,pressed_=0;
 public:
  KeyEvent update(bool pressed,uint32_t ms) {
    if (pressed!=raw_) { raw_=pressed;change_=ms; }
    if (ms-change_>=25 && down_!=raw_) {
      down_=raw_;
      if (down_) { pressed_=ms;long_=false; }
      else if (!long_) return KeyEvent::Short;
    }
    if (down_&&!long_&&ms-pressed_>=1200) { long_=true;return KeyEvent::Long; }
    return KeyEvent::None;
  }
};
inline int amplitude_pixel(float uV,float range,int centre,int half) {
  return centre-static_cast<int>(std::lround(std::clamp(uV/range,-1.f,1.f)*half));
}
}
