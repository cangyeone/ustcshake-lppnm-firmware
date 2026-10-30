#include "core.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace shake;
float fromBits(uint32_t x) {float f;std::memcpy(&f,&x,4);return f;}
int main() {
  Settings s;assert(s.valid());s.highpass=10;assert(!s.valid());s=Settings{};
  Bandpass f;f.configure(s);
  // Coefficients read from original firmware's running RAM on 2026-10-30.
  uint32_t expected[10]={0x3f7dbc3f,0xbffdbc3f,0x3f7dbc3f,0xbffdb9ae,0x3f7b7d9e,0x3c5ae08a,0x3cdae08a,0x3c5ae08a,0xbfd2dff8,0x3f336df9};
  float actual[10]={f.hp.b0,f.hp.b1,f.hp.b2,f.hp.a1,f.hp.a2,f.lp.b0,f.lp.b1,f.lp.b2,f.lp.a1,f.lp.a2};
  for(int i=0;i<10;i++) assert(std::abs(actual[i]-fromBits(expected[i]))<2e-7);
  f.configure(s,1000);for(int i=0;i<10000;i++) assert(std::abs(f.push(1000))<1e-5);
  for(double freq:{5.,25.,75.,110.}) {
    Resampler r;double energy=0,error=0;int count=0;int64_t prev=0;
    for(int i=0;i<10000;i++) {
      float y;int64_t t;
      if(r.push(std::sin(2*M_PI*freq*i/250),y,t)) {
        if(prev) assert(t-prev==10000);else assert(t==160000);prev=t;
        if(i>1000) {energy+=y*y;error=std::max(error,std::abs(y-std::sin(2*M_PI*freq*t/1e6)));count++;}
      }
    }
    if(freq<=25) {assert(std::abs(energy/count-.5)<.002);assert(error<.001);}
    else assert(energy/count<1e-7);
    r.reset();float y;int64_t t;for(int i=0;i<80;i++) assert(!r.push(1,y,t));assert(r.push(1,y,t));assert(t==160000&&std::abs(y-1)<.0001);
  }
  PeakDetector detector;std::vector<Pick> events;
  std::array<float,7> p{};int64_t t=0;
  auto step=[&](float a,float b){p[1]=a;p[2]=b;detector.push(t,p,.3f,[&](Pick v){events.push_back(v);});t+=80000;};
  step(.4,.1);step(.8,.1);for(int i=0;i<5;i++) step(.2,.1);
  assert(events.size()==1&&events[0].phase==1&&events[0].time_us==80000);
  step(.7,.9);for(int i=0;i<5;i++) step(.1,.1);
  assert(events.size()==2&&events[1].phase==2); // Independent S accepted; duplicate P suppressed.
  detector.reset();events.clear();t=0;step(.8,.9);for(int i=0;i<5;i++) step(.1,.1);assert(events.size()==2);
  Button b;assert(b.update(true,100)==KeyEvent::None);assert(b.update(false,110)==KeyEvent::None);
  assert(b.update(true,200)==KeyEvent::None);assert(b.update(true,225)==KeyEvent::None);
  assert(b.update(false,300)==KeyEvent::None);assert(b.update(false,325)==KeyEvent::Short);
  b.update(true,400);b.update(true,425);assert(b.update(true,1625)==KeyEvent::Long);b.update(false,1700);assert(b.update(false,1725)==KeyEvent::None);
  assert(amplitude_pixel(100,50,192,177)==15);assert(amplitude_pixel(-100,50,192,177)==369);
  puts("PASS: original filter coefficients, FIR rate/timestamp/alias rejection, phase confirmation/dedup/reset, keys and clipping");
}
