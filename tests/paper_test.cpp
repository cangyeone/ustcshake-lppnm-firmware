#include "paper.hpp"
#include <cassert>
#include <fstream>
#include <vector>
using namespace shake;
int main(int argc,char** argv) {
  PaperAnnotations labels;
  std::vector<uint8_t> page;
  int previous=246;
  for(int y=0;y<650;y++) {
    uint8_t row[48]{};
    float value=20*std::sin(y*.17f)+110*std::exp(-std::pow((y-125)/35.f,2))*std::sin(y*.33f)+70*std::exp(-std::pow((y-300)/40.f,2))*std::sin(y*.25f);
    int x=amplitude_pixel(value,150,246,118);
    for(int k=std::min(x,previous);k<=std::max(x,previous);k++) { paperPixel(row,k); }
    previous=x;
    paperPixel(row,126);paperPixel(row,366);if(y%5==0) paperPixel(row,246);
    if(y%125==0) for(int k=126;k<=366;k+=4) paperPixel(row,k);
    if(y==125) labels.add({1'000'000,.123f,1},1'793'361'600'000'000LL);
    if(y==300) labels.add({2'400'000,.042f,2},1'793'361'600'000'000LL);
    if(y==450) {labels.add({3'600'000,.019f,1},1'793'361'600'000'000LL);labels.add({3'600'000,.024f,2},1'793'361'600'000'000LL);}
    if(y==649) labels.add({5'192'000,.018f,1},1'793'361'600'000'000LL);
    labels.render(row);
    if(y==125) assert(row[124/8]&(0x80>>(124%8)));
    page.insert(page.end(),row,row+48);
  }
  while(labels.pending()) {uint8_t row[48]{};labels.render(row);page.insert(page.end(),row,row+48);}
  assert(labels.markers==5&&labels.dropped==0);
  assert(page.size()/48==697); // End-of-record label fully printed.
  if(argc>1) {std::ofstream f(argv[1],std::ios::binary);f<<"P4\n384 "<<page.size()/48<<"\n";f.write(reinterpret_cast<const char*>(page.data()),page.size());}
  puts("PASS: arrival-row markers, coincident P/S, stacked margin labels, end-of-record labels");
}
