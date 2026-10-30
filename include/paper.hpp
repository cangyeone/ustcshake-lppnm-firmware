#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include "core.hpp"
namespace shake {
inline void paperPixel(uint8_t* row,int x) {if(x>=0&&x<384) row[x/8]|=uint8_t(0x80>>(x%8));}
inline uint8_t glyphRow(char c,int y) {
  static constexpr uint8_t digits[10][7]={
    {14,17,19,21,25,17,14},{4,12,4,4,4,4,14},{14,17,1,2,4,8,31},
    {30,1,1,14,1,1,30},{2,6,10,18,31,2,2},{31,16,16,30,1,1,30},
    {14,16,16,30,17,17,14},{31,1,2,4,8,8,8},{14,17,17,14,17,17,14},
    {14,17,17,15,1,1,14}};
  static constexpr uint8_t p[]={30,17,17,30,16,16,16},s[]={15,16,16,14,1,1,30};
  static constexpr uint8_t u[]={17,17,17,17,17,17,14},t[]={31,4,4,4,4,4,4},cC[]={15,16,16,16,16,16,15};
  static constexpr uint8_t b[]={30,17,17,30,17,17,30},o[]={14,17,17,17,17,17,14};
  if(y<0||y>=7) return 0;
  if(c>='0'&&c<='9') return digits[c-'0'][y];
  switch(c) {
    case 'P':return p[y];case 'S':return s[y];case 'U':return u[y];case 'T':return t[y];case 'C':return cC[y];case 'B':return b[y];case 'O':return o[y];
    case '.':return y==6?4:0;case ':':return (y==2||y==5)?4:0;case '-':return y==3?14:0;case '+':return y==3?14:(y>=1&&y<=5?4:0);default:return 0;
  }
}
inline void paperTextRow(uint8_t* row,const char* text,int y,int x,int scale=2) {
  if(y<0||y>=7*scale) return;
  for(int i=0;text[i];i++) {
    uint8_t bits=glyphRow(text[i],y/scale);
    for(int k=0;k<5;k++) if(bits&(1<<(4-k))) for(int dx=0;dx<scale;dx++) paperPixel(row,x+i*6*scale+k*scale+dx);
  }
}
// Labels use the left margin and never insert rows into the waveform time axis.
// Closely spaced labels stack; the dashed line and P/S glyph stay at the exact
// arrival row. Independent P/S marker columns allow coincident P and S.
class PaperAnnotations {
  struct Label { char text[3][12]{}; };
  std::array<Label,8> labels{};
  int head=0,count=0,labelRow=0;
  int markerRemaining[2]{};
  bool markLine=false;
 public:
  uint32_t markers=0,dropped=0;
  bool pending() const {return count||markerRemaining[0]||markerRemaining[1];}
  void add(const Pick& p,int64_t offset) {
    if(p.phase<1||p.phase>2) return;
    markerRemaining[p.phase-1]=14;markLine=true;markers++;
    if(count==int(labels.size())) {dropped++;return;}
    auto& label=labels[(head+count++)%labels.size()];
    snprintf(label.text[0],sizeof(label.text[0]),"%c %.3f",p.phase==1?'P':'S',p.score);
    if(offset) {
      int64_t ms=(p.time_us+offset)/1000;time_t sec=ms/1000;struct tm t{};gmtime_r(&sec,&t);
      strftime(label.text[1],sizeof(label.text[1]),"%H:%M:%S",&t);
      snprintf(label.text[2],sizeof(label.text[2]),".%03d UTC",int(ms%1000));
    } else {
      snprintf(label.text[1],sizeof(label.text[1]),"%.2f",double(p.time_us)/1e6);
      snprintf(label.text[2],sizeof(label.text[2]),"BOOT");
    }
  }
  void render(uint8_t* row) {
    if(markLine) {for(int x=124;x<372;x+=4) paperPixel(row,x);markLine=false;}
    for(int i=0;i<2;i++) if(markerRemaining[i]) {
      paperTextRow(row,i==0?"P":"S",14-markerRemaining[i],i==0?110:373);
      markerRemaining[i]--;
    }
    if(count) {
      paperTextRow(row,labels[head].text[labelRow/16],labelRow%16,4);
      if(++labelRow==48) {labelRow=0;head=(head+1)%labels.size();count--;}
    }
  }
};
}
