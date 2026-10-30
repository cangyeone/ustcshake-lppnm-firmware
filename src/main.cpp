#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <WiFi.h>
#include <WebServer.h>
#include <U8g2lib.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <atomic>
#include <ctime>
#include "board.hpp"
#include "core.hpp"
#include "paper.hpp"
#include "../model/artifacts/model.hpp"

using shake::Settings;
using shake::Pick;
constexpr char version[]="v2.0.0-lppnm-dev";
constexpr size_t historySize=2500, pickCapacity=128;
struct Sample { int64_t us; uint32_t seq; float uv; };
struct PaperRow { int64_t us; float lo,hi; };
struct Metrics {
  uint32_t acquired=0, missed=0, invalid=0, saturated=0, nnDropped=0, nnResets=0;
  uint32_t inferenceUs=0,inferenceMaxUs=0,inferenceBlocks=0,nnQueueMax=0;
  uint32_t adcLateMaxUs=0,adcReadMaxUs=0,usbDropped=0,paperOverflow=0;
  uint32_t printedRows=0,printerBytes=0,displayMaxUs=0,pickCount=0;
  int64_t watermark=0,coverageStart=0;
  uint32_t paperMarkers=0,paperLabelsDropped=0;
  uint32_t modelSamples=0;float pProbability=0,sProbability=0;
};
struct PrintJob {
  bool active=false,ended=false,dry=false,cancel=false,overflow=false;
  int64_t start=0,end=0,epochOffset=0;
  uint32_t nnResets=0,invalid=0,missed=0;
  float range=50;
  Settings settings;
};
portMUX_TYPE mux=portMUX_INITIALIZER_UNLOCKED;
Metrics metrics;
Settings settings;
uint32_t settingsRevision=1;
PrintJob job;
float history[historySize]{};
uint32_t historyCount=0;
int64_t historyLastUs=0;
Pick picks[pickCapacity]{};
uint32_t pickCount=0;
float autoRange=50;
int64_t epochOffset=0;
QueueHandle_t inferenceQueue,usbQueue,paperQueue;
TaskHandle_t acquisitionHandle,inferenceHandle,paperHandle;
esp_timer_handle_t sampleTimer;
int64_t samplingOrigin=0;
std::atomic<bool> usbStreaming{false};
std::atomic<bool> adcReady{false};
HardwareSerial printer(1);
WebServer web(80);
// Use the library's horizontal framebuffer and fonts. The hardware transfer
// below preserves the original SH1122 power/contrast sequence and 8 MHz SPI.
U8G2_SH1122_256X64_F_4W_HW_SPI oled(U8G2_R0,board::oled_cs,board::oled_dc);
lppnm::OnlinePicker* picker=nullptr;
char ssid[40];
bool menu=false,editing=false,exitMenu=false;
int menuRow=0,exitChoice=0;
Settings pending;
char notice[80]{};
uint32_t noticeUntil=0;

Metrics metricsCopy() { portENTER_CRITICAL(&mux);auto x=metrics;portEXIT_CRITICAL(&mux);return x; }
PrintJob jobCopy() { portENTER_CRITICAL(&mux);auto x=job;portEXIT_CRITICAL(&mux);return x; }
Settings settingsCopy() { portENTER_CRITICAL(&mux);auto x=settings;portEXIT_CRITICAL(&mux);return x; }
void message(const char* s) { snprintf(notice,sizeof(notice),"%s",s);noticeUntil=millis()+3000; }
void publishPick(const Pick& p) {
  portENTER_CRITICAL(&mux);
  picks[pickCount++%pickCapacity]=p;metrics.pickCount=pickCount;
  portEXIT_CRITICAL(&mux);
}
void clockString(char* out,size_t n,int64_t us,int64_t offset,bool date=false) {
  if (!offset) { snprintf(out,n,"BOOT+%.3fs",double(us)/1e6);return; }
  int64_t ms=(us+offset)/1000;
  time_t sec=ms/1000;struct tm t{};gmtime_r(&sec,&t);
  char tmp[40];strftime(tmp,sizeof(tmp),date?"%Y-%m-%d %H:%M:%S":"%H:%M:%S",&t);
  snprintf(out,n,"%s.%03d UTC",tmp,int(ms%1000));
}
bool writeAdc(uint8_t reg,uint16_t value) {
  Wire.beginTransmission(board::adc_address);Wire.write(reg);
  Wire.write(uint8_t(value>>8));Wire.write(uint8_t(value));return Wire.endTransmission()==0;
}
bool readAdc(uint8_t reg,int16_t& value) {
  Wire.beginTransmission(board::adc_address);Wire.write(reg);
  if (Wire.endTransmission(false)!=0) return false;
  if (Wire.requestFrom(uint8_t(board::adc_address),size_t(2),true)!=2) return false;
  value=static_cast<int16_t>((Wire.read()<<8)|Wire.read());return true;
}
void tick(void*) { xTaskNotifyGive(acquisitionHandle); }

void acquire(void*) {
  shake::Bandpass filter;
  uint32_t revision=0,lastSeq=0;bool first=true,haveBaseline=false;
  float pairLo=0,pairHi=0;int pairCount=0;
  for (;;) {
    ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
    int64_t now=esp_timer_get_time();
    uint32_t seq=static_cast<uint32_t>((now-samplingOrigin)/4000);
    if (!first && seq==lastSeq) continue;
    uint32_t lost=first?0:seq-lastSeq-1;first=false;lastSeq=seq;
    int64_t scheduled=samplingOrigin+int64_t(seq)*4000;
    auto cfg=settingsCopy();
    portENTER_CRITICAL(&mux);uint32_t rev=settingsRevision;portEXIT_CRITICAL(&mux);
    if (revision!=rev) { filter.configure(cfg);revision=rev;haveBaseline=false; }
    int16_t raw=0;int64_t before=esp_timer_get_time();bool ok=readAdc(0,raw);
    uint32_t cost=esp_timer_get_time()-before;
    portENTER_CRITICAL(&mux);
    metrics.missed+=lost;metrics.invalid+=!ok;
    metrics.adcLateMaxUs=std::max(metrics.adcLateMaxUs,uint32_t(now-scheduled));
    metrics.adcReadMaxUs=std::max(metrics.adcReadMaxUs,cost);
    if (job.active && scheduled>=job.end) job.ended=true;
    portEXIT_CRITICAL(&mux);
    if (lost||!ok) {
      portENTER_CRITICAL(&mux);
      for(uint32_t k=0;k<std::min<uint32_t>(lost+!ok,historySize);k++) history[historyCount++%historySize]=NAN;
      historyLastUs=scheduled;portEXIT_CRITICAL(&mux);
    }
    if (!ok) { adcReady=false;haveBaseline=false;continue; }
    adcReady=true;
    float uv=float(raw)*125.f;
    if (!haveBaseline || lost) { filter.configure(cfg,uv);haveBaseline=true; }
    float filtered=filter.push(uv);
    Sample sample{scheduled,seq,uv};
    if (xQueueSend(inferenceQueue,&sample,0)!=pdTRUE) {
      portENTER_CRITICAL(&mux);metrics.nnDropped++;portEXIT_CRITICAL(&mux);
    }
    if (usbStreaming && xQueueSend(usbQueue,&sample,0)!=pdTRUE) {
      portENTER_CRITICAL(&mux);metrics.usbDropped++;portEXIT_CRITICAL(&mux);
    }
    portENTER_CRITICAL(&mux);
    history[historyCount++%historySize]=filtered;historyLastUs=scheduled;
    metrics.acquired++;metrics.saturated+=(raw==INT16_MIN||raw==INT16_MAX);
    bool printing=job.active&&!job.ended&&!job.cancel&&!job.overflow&&scheduled>=job.start;
    portEXIT_CRITICAL(&mux);
    if (printing) {
      if (!pairCount) pairLo=pairHi=filtered;
      else { pairLo=std::min(pairLo,filtered);pairHi=std::max(pairHi,filtered); }
      if (++pairCount==2) {
        PaperRow row{scheduled,pairLo,pairHi};pairCount=0;
        if (xQueueSend(paperQueue,&row,0)!=pdTRUE) {
          portENTER_CRITICAL(&mux);metrics.paperOverflow++;job.overflow=true;job.ended=true;portEXIT_CRITICAL(&mux);
        }
      }
    } else pairCount=0;
  }
}

void infer(void*) {
  shake::Resampler resampler;shake::PeakDetector detector;
  float block[128];lppnm::RawOutput output[lppnm::kMaxOutput];
  int fill=0;int64_t origin=0,modelOrigin=0;uint32_t expected=0;bool first=true;
  Sample s;
  for (;;) {
    if (xQueueReceive(inferenceQueue,&s,portMAX_DELAY)!=pdTRUE) continue;
    bool gap=!first&&s.seq!=expected;
    if (first||gap) {
      picker->reset();resampler.reset();detector.reset();fill=0;
      origin=s.us;modelOrigin=0;
      portENTER_CRITICAL(&mux);
      metrics.nnResets+=gap;metrics.watermark=metrics.coverageStart=0;
      portEXIT_CRITICAL(&mux);
    }
    first=false;expected=s.seq+1;
    float y;int64_t relative;
    if (!resampler.push(s.uv,y,relative)) continue;
    if (!modelOrigin) modelOrigin=origin+relative;
    block[fill++]=y;
    if (fill<128) continue;
    fill=0;int n=0;int64_t before=esp_timer_get_time();
    bool ok=picker->push(block,output,n);
    auto cfg=settingsCopy();
    int64_t firstOutput=0,lastOutput=0;float pMax=0,sMax=0;
    for (int i=0;ok&&i<n;i++) {
      auto p=picker->engine.decode(output[i]);
      pMax=std::max(pMax,p.probability[1]);sMax=std::max(sMax,p.probability[2]);
      int64_t time=modelOrigin+int64_t(std::llround(p.sample*10000.0));
      if (!i) firstOutput=time;
      lastOutput=modelOrigin+p.bin*80000;
      detector.push(time,p.probability,cfg.threshold,publishPick);
    }
    uint32_t elapsed=esp_timer_get_time()-before;
    auto queued=uxQueueMessagesWaiting(inferenceQueue);
    portENTER_CRITICAL(&mux);
    metrics.inferenceUs=elapsed;metrics.inferenceMaxUs=std::max(metrics.inferenceMaxUs,elapsed);
    metrics.modelSamples=picker->normalizer.count;metrics.pProbability=pMax;metrics.sProbability=sMax;
    metrics.inferenceBlocks++;metrics.nnQueueMax=std::max(metrics.nnQueueMax,uint32_t(queued));
    if (n) {
      if (!metrics.coverageStart) metrics.coverageStart=firstOutput;
      metrics.watermark=lastOutput-400000; // Peaks still awaiting confirmation are excluded.
    }
    portEXIT_CRITICAL(&mux);
    if (!ok) first=true;
    // Let the idle task and USB/UI run even during queue recovery.
    vTaskDelay(1);
  }
}

void oledCommands(const uint8_t* bytes,size_t count) {
  SPI.beginTransaction(SPISettings(board::oled_hz,MSBFIRST,SPI_MODE0));
  digitalWrite(board::oled_cs,LOW);digitalWrite(board::oled_dc,LOW);
  SPI.writeBytes(bytes,count);digitalWrite(board::oled_cs,HIGH);SPI.endTransaction();
}
void initializeDisplay() {
  pinMode(board::oled_cs,OUTPUT);pinMode(board::oled_dc,OUTPUT);
  digitalWrite(board::oled_cs,HIGH);digitalWrite(board::oled_dc,HIGH);
  SPI.begin(board::oled_sck,-1,board::oled_mosi,board::oled_cs);
  delay(150);
  // Exact original panel power settings; start-line 32 and hardware remap retained.
  const uint8_t init[]={0xae,0xb0,0x00,0x10,0x00,0xd5,0x50,0xd9,0x22,0x60,
    0x81,0xff,0xa1,0xc8,0xd3,0x00,0xa4,0xa6,0xa8,0x3f,0xad,0x80,0xdb,0x30,0xdc,0x30,0x33,0xaf};
  oledCommands(init,sizeof(init));oled.clearBuffer();oled.setFont(u8g2_font_wqy12_t_gb2312);
}
void flushDisplay() {
  int64_t before=esp_timer_get_time();
  const uint8_t* mono=oled.getBufferPtr();uint8_t row[128];
  // SH1122/U8g2 stores rows as MSB-first monochrome bytes, as in the original.
  for (int y=0;y<64;y++) {
    for (int x=0;x<128;x++) {
      uint8_t v=mono[y*32+x/4];int shift=6-(x%4)*2;
      row[x]=((v>>(shift+1))&1?0x80:0)|((v>>shift)&1?0x08:0);
    }
    SPI.beginTransaction(SPISettings(board::oled_hz,MSBFIRST,SPI_MODE0));
    digitalWrite(board::oled_cs,LOW);digitalWrite(board::oled_dc,LOW);
    SPI.transfer(0xb0);SPI.transfer(y);SPI.transfer(0x10);SPI.transfer(0x00);
    digitalWrite(board::oled_dc,HIGH);SPI.writeBytes(row,sizeof(row));
    digitalWrite(board::oled_cs,HIGH);SPI.endTransaction();
  }
  uint32_t elapsed=esp_timer_get_time()-before;
  portENTER_CRITICAL(&mux);metrics.displayMaxUs=std::max(metrics.displayMaxUs,elapsed);portEXIT_CRITICAL(&mux);
}

bool paperWrite(const uint8_t* data,size_t size,bool dry) {
  if (dry) { vTaskDelay(pdMS_TO_TICKS((size*10000/board::printer_baud)+1));return true; }
  int64_t limit=esp_timer_get_time()+1'000'000;
  while (size) {
    if (esp_timer_get_time()>limit) return false; // Finish one raster command before honoring cancellation.
    size_t chunk=std::min(size,size_t(std::max(printer.availableForWrite(),0)));
    if (!chunk) { vTaskDelay(1);continue; }
    auto sent=printer.write(data,chunk);data+=sent;size-=sent;
    portENTER_CRITICAL(&mux);metrics.printerBytes+=sent;portEXIT_CRITICAL(&mux);
  }
  return true;
}
bool paperText(const char* s,bool dry) { return paperWrite(reinterpret_cast<const uint8_t*>(s),strlen(s),dry); }
void pixel(uint8_t* row,int x) { if(x>=0&&x<384) row[8+x/8]|=uint8_t(0x80>>(x%8)); }
void paperTask(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
    auto j=jobCopy();if (!j.active) continue;
    char text[240],stamp[64];clockString(stamp,sizeof(stamp),j.start,j.epochOffset,true);
    const uint8_t initialize[]={0x1b,0x40};bool ok=paperWrite(initialize,sizeof(initialize),j.dry);
    if (j.settings.logo) ok=ok&&paperText("       /\\    USTCShake\n   ___/  \\___ SEISMOGRAM\n",j.dry);
    snprintf(text,sizeof(text),"%s\n%s  duration:%us\nBand %.2f-%.2f Hz\nRange +/-%.3f %s\nLPPNM INT8 3072/128\nP/S marked on waveform; buffer 6s\n\n",
      version,stamp,j.settings.duration,j.settings.highpass,j.settings.lowpass,
      j.settings.velocity?j.range/(1000*j.settings.sensitivity):j.range/1000,
      j.settings.velocity?"mm/s":"mV");
    ok=ok&&paperText(text,j.dry);
    uint32_t rows=0;int previous=246;PaperRow value;
    shake::PaperAnnotations annotations;
    Pick copy[pickCapacity];uint32_t count=0,cachedCount=UINT32_MAX;
    for (;;) {
      auto live=jobCopy();if (live.cancel||live.overflow||!ok) break;
      if (xQueueReceive(paperQueue,&value,pdMS_TO_TICKS(10))==pdTRUE) {
        // ADC continues to fill a bounded queue while the model confirms this row.
        // Six seconds covers FIR, noncausal CNN lookahead and peak confirmation.
        while(esp_timer_get_time()<value.us+6'000'000 && !jobCopy().cancel && !jobCopy().overflow) vTaskDelay(pdMS_TO_TICKS(10));
        if(jobCopy().cancel||jobCopy().overflow) break;
        portENTER_CRITICAL(&mux);
        if(cachedCount!=pickCount) {
          cachedCount=pickCount;count=std::min(pickCount,uint32_t(pickCapacity));
          for(uint32_t i=0;i<count;i++) copy[i]=picks[(pickCount-count+i)%pickCapacity];
        }
        portEXIT_CRITICAL(&mux);
        for(uint32_t i=0;i<count;i++) if(copy[i].time_us>value.us-8000 && copy[i].time_us<=value.us && copy[i].time_us>=j.start) annotations.add(copy[i],j.epochOffset);
        uint8_t raster[56]={0x1d,0x76,0x30,0x00,48,0,1,0};
        int a=shake::amplitude_pixel(value.hi,j.range,246,118);
        int b=shake::amplitude_pixel(value.lo,j.range,246,118);
        // Connect extrema from adjacent 8 ms intervals without hiding short impulses.
        a=std::min(a,previous);b=std::max(b,previous);
        for (int x=a;x<=b;x++) pixel(raster,x);
        previous=shake::amplitude_pixel((value.hi+value.lo)*.5f,j.range,246,118);
        pixel(raster,126);pixel(raster,366);if(rows%5==0) pixel(raster,246);
        if (rows%125==0) for(int x=126;x<=366;x+=4) pixel(raster,x);
        annotations.render(raster+8);
        ok=paperWrite(raster,sizeof(raster),j.dry);rows++;
        portENTER_CRITICAL(&mux);metrics.printedRows++;metrics.paperMarkers=annotations.markers;metrics.paperLabelsDropped=annotations.dropped;portEXIT_CRITICAL(&mux);
      } else if (live.ended) break;
    }
    // Finish margin labels for arrivals near the end; no artificial waveform data.
    while(ok&&!jobCopy().cancel&&annotations.pending()) {
      uint8_t raster[56]={0x1d,0x76,0x30,0x00,48,0,1,0};annotations.render(raster+8);ok=paperWrite(raster,sizeof(raster),j.dry);
    }
    if (jobCopy().cancel) { paperText("\nCANCELLED\n",j.dry); }
    else {
      // The waveform has a six-second queue. The footer also checks coverage
      // and explicitly reports warmup or gaps instead of inventing arrivals.
      int64_t deadline=esp_timer_get_time()+8'000'000;
      while (ok&&!jobCopy().overflow&&metricsCopy().watermark<j.end&&esp_timer_get_time()<deadline&&!jobCopy().cancel) vTaskDelay(pdMS_TO_TICKS(50));
      paperText("\n--- P / S ARRIVALS ---\n",j.dry);
      portENTER_CRITICAL(&mux);count=std::min(pickCount,uint32_t(pickCapacity));
      for(uint32_t i=0;i<count;i++) copy[i]=picks[(pickCount-count+i)%pickCapacity];portEXIT_CRITICAL(&mux);
      int found=0;
      for(uint32_t i=0;i<count;i++) if(copy[i].time_us>=j.start&&copy[i].time_us<j.end) {
        auto p=copy[i];clockString(stamp,sizeof(stamp),p.time_us,j.epochOffset);
        snprintf(text,sizeof(text),"%c  %s\n  +%.3fs  score=%.3f\n",p.phase==1?'P':'S',stamp,double(p.time_us-j.start)/1e6,p.score);
        ok=paperText(text,j.dry)&&ok;found++;
      }
      auto m=metricsCopy();
      bool covered=m.coverageStart&&m.coverageStart<=j.start&&m.watermark>=j.end&&m.nnResets==j.nnResets;
      if (!found) paperText("No confirmed P/S picks in interval\n",j.dry);
      if (!covered) paperText("INCOMPLETE MODEL COVERAGE (warmup/gap)\n",j.dry);
      if(annotations.dropped) paperText("Some margin labels omitted; see arrival list\n",j.dry);
      if (m.invalid!=j.invalid||m.missed!=j.missed||jobCopy().overflow||!ok) paperText("WARNING: recording/transport gap\n",j.dry);
      snprintf(text,sizeof(text),"Rows:%lu  threshold:%.2f\nTimes are model estimates\n%s\n\n\n",(unsigned long)rows,j.settings.threshold,j.epochOffset?"Clock: synchronized":"Clock: UNSYNCED, boot-relative");paperText(text,j.dry);
    }
    portENTER_CRITICAL(&mux);job.active=false;portEXIT_CRITICAL(&mux);
  }
}
bool startPrint(bool dry=false,int duration=0) {
  if (!adcReady) { message("采集异常，不能打印");return false; }
  portENTER_CRITICAL(&mux);bool active=job.active;portEXIT_CRITICAL(&mux);
  if (active) { message("正在打印");return false; }
  xQueueReset(paperQueue);
  portENTER_CRITICAL(&mux);
  job={};job.settings=settings;if(duration) job.settings.duration=duration;
  job.start=esp_timer_get_time();job.end=job.start+int64_t(job.settings.duration)*1'000'000;
  job.range=settings.range?shake::ranges[settings.range-1]:autoRange;
  job.epochOffset=epochOffset;job.nnResets=metrics.nnResets;
  job.invalid=metrics.invalid;job.missed=metrics.missed;job.dry=dry;job.active=true;
  portEXIT_CRITICAL(&mux);xTaskNotifyGive(paperHandle);return true;
}

void hello() {
  Serial.printf("{\"type\":\"hello\",\"protocol\":\"espshake-usb\",\"version\":1,\"session\":1,\"device_id\":\"%012llX\",\"software\":\"%s\",\"hardware\":\"v2.0\",\"signal\":\"raw_adc\",\"sample_rate_hz\":250,\"unit\":\"uV\",\"sensitivity_mv_per_mm_s\":%.7f,\"display_band_hz\":[%.2f,%.2f]}\n",
    ESP.getEfuseMac(),version,settings.sensitivity,settings.highpass,settings.lowpass);
}
void status() {
  auto m=metricsCopy();auto j=jobCopy();auto cfg=settingsCopy();
  Serial.printf("{\"type\":\"status\",\"firmware\":\"%s\",\"adc_ready\":%s,\"sample_rate_hz\":250,\"model_rate_hz\":100,\"window\":3072,\"hop\":128,\"quantization\":\"W8A8-fused\",\"acquired\":%lu,\"missed_adc\":%lu,\"invalid_adc\":%lu,\"nn_dropped\":%lu,\"nn_resets\":%lu,\"inference_us\":%lu,\"inference_max_us\":%lu,\"inference_blocks\":%lu,\"nn_queue_max\":%lu,\"adc_late_max_us\":%lu,\"adc_read_max_us\":%lu,\"display_max_us\":%lu,\"paper_overflow\":%lu,\"printed_rows\":%lu,\"printer_bytes\":%lu,\"printing\":%s,\"dry_run\":%s,\"picks\":%lu,\"watermark_us\":%lld,\"heap_free\":%u,\"heap_min\":%u,\"range_index\":%u,\"filter_hz\":[%.2f,%.2f],\"key_raw\":[%d,%d,%d]}\n",
    version,adcReady?"true":"false",(unsigned long)m.acquired,(unsigned long)m.missed,(unsigned long)m.invalid,(unsigned long)m.nnDropped,(unsigned long)m.nnResets,(unsigned long)m.inferenceUs,(unsigned long)m.inferenceMaxUs,(unsigned long)m.inferenceBlocks,(unsigned long)m.nnQueueMax,(unsigned long)m.adcLateMaxUs,(unsigned long)m.adcReadMaxUs,(unsigned long)m.displayMaxUs,(unsigned long)m.paperOverflow,(unsigned long)m.printedRows,(unsigned long)m.printerBytes,j.active?"true":"false",j.dry?"true":"false",(unsigned long)m.pickCount,(long long)m.watermark,ESP.getFreeHeap(),ESP.getMinFreeHeap(),cfg.range,cfg.highpass,cfg.lowpass,digitalRead(board::key_left),digitalRead(board::key_center),digitalRead(board::key_right));
  Serial.printf("{\"type\":\"printer\",\"buffer_seconds\":6,\"phase_markers\":%lu,\"labels_dropped\":%lu}\n",(unsigned long)m.paperMarkers,(unsigned long)m.paperLabelsDropped);
  Serial.printf("{\"type\":\"inference\",\"model_samples\":%lu,\"p_probability\":%.6f,\"s_probability\":%.6f,\"threshold\":%.2f,\"stack_min_bytes\":[%u,%u,%u]}\n",(unsigned long)m.modelSamples,m.pProbability,m.sProbability,cfg.threshold,uxTaskGetStackHighWaterMark(acquisitionHandle),uxTaskGetStackHighWaterMark(inferenceHandle),uxTaskGetStackHighWaterMark(paperHandle));
}
void setEpoch(int64_t ms) {
  if(ms<1577836800000LL||ms>4102444800000LL) return;
  portENTER_CRITICAL(&mux);epochOffset=ms*1000-esp_timer_get_time();portEXIT_CRITICAL(&mux);
}
void serialPoll() {
  static char command[120];static size_t used=0;static uint32_t lastStream=0;
  while(Serial.available()) {
    char c=Serial.read();if(c=='\r') continue;
    if(c=='\n') {
      command[used]=0;used=0;
      if(!strcmp(command,"stream on")) {xQueueReset(usbQueue);usbStreaming=true;hello();}
      else if(!strcmp(command,"stream off")) usbStreaming=false;
      else if(!strcmp(command,"status")||!strcmp(command,"stream status")) status();
      else if(!strcmp(command,"hello")) hello();
      else if(!strcmp(command,"cancel")) {portENTER_CRITICAL(&mux);job.cancel=true;portEXIT_CRITICAL(&mux);}
      else if(!strncmp(command,"time ",5)) setEpoch(strtoll(command+5,nullptr,10));
      else if(!strncmp(command,"dryrun ",7)) {int d=atoi(command+7);if(d>=5&&d<=60) startPrint(true,d);}
      else if(!strcmp(command,"mode auto")||!strcmp(command,"mode fixed")||!strcmp(command,"mode toggle")) {
        if(!jobCopy().active) {portENTER_CRITICAL(&mux);settings.range=!strcmp(command,"mode auto")?0:!strcmp(command,"mode fixed")?1:settings.range?0:1;portEXIT_CRITICAL(&mux);}
      }
      else if(!strcmp(command,"picks")) {
        Pick copy[pickCapacity];uint32_t n;
        portENTER_CRITICAL(&mux);n=std::min(pickCount,uint32_t(pickCapacity));for(uint32_t i=0;i<n;i++) copy[i]=picks[(pickCount-n+i)%pickCapacity];portEXIT_CRITICAL(&mux);
        for(uint32_t i=0;i<n;i++) Serial.printf("{\"type\":\"pick\",\"phase\":\"%c\",\"device_time_us\":%lld,\"score\":%.4f}\n",copy[i].phase==1?'P':'S',(long long)copy[i].time_us,copy[i].score);
      }
    } else if(used+1<sizeof(command)) command[used++]=c;
    else used=0;
  }
  if(usbStreaming&&millis()-lastStream>=32) {
    lastStream=millis();Sample s;int n=0;
    if(Serial.availableForWrite()<1500) return;
    Serial.print("{\"type\":\"samples\",\"session\":1,\"data\":[");
    while(n<24&&xQueueReceive(usbQueue,&s,0)==pdTRUE) {
      Serial.printf("%s[%lu,%lld,%.0f,0]",n?",":"",(unsigned long)s.seq,(long long)s.us,s.uv);n++;
    }
    Serial.println("]}");
  }
}

void adjust(int direction) {
  switch(menuRow) {
    case 0: pending.duration=std::clamp(int(pending.duration)+direction*5,5,60);break;
    case 1: pending.range=(pending.range+direction+12)%12;break;
    case 2: pending.highpass=std::clamp(pending.highpass+direction*.1f,.1f,20.f);break;
    case 3: pending.lowpass=std::clamp(pending.lowpass+direction*1.f,1.f,100.f);break;
    case 4: pending.velocity=!pending.velocity;break;
    case 5: pending.logo=!pending.logo;break;
    case 6: pending.threshold=std::clamp(pending.threshold+direction*.01f,.01f,.95f);break;
  }
}
void buttons() {
  static shake::Button left,center,right;
  uint32_t ms=millis();auto l=left.update(!digitalRead(board::key_left),ms);
  auto c=center.update(!digitalRead(board::key_center),ms);
  auto r=right.update(!digitalRead(board::key_right),ms);
  int move=(r==shake::KeyEvent::Short)-(l==shake::KeyEvent::Short);
  if(c==shake::KeyEvent::Long) {
    if(!menu) {pending=settingsCopy();menu=true;editing=false;exitMenu=false;menuRow=0;}
    else if(editing) editing=false;
    else {exitMenu=true;exitChoice=0;}
    return;
  }
  if(!menu) {
    if(move&&!jobCopy().active) {portENTER_CRITICAL(&mux);settings.range=(settings.range+move+12)%12;portEXIT_CRITICAL(&mux);}
    if(c==shake::KeyEvent::Short) startPrint();
    return;
  }
  if(exitMenu) {
    if(move) exitChoice=(exitChoice+move+3)%3;
    if(c==shake::KeyEvent::Short) {
      if(exitChoice==2) exitMenu=false;
      else if(exitChoice==1) {menu=false;exitMenu=false;}
      else if(!pending.valid()) message("高通须小于低通");
      else if(jobCopy().active) message("打印后可应用设置");
      else {portENTER_CRITICAL(&mux);settings=pending;settingsRevision++;portEXIT_CRITICAL(&mux);menu=false;exitMenu=false;message("设置仅本次开机有效");}
    }
    return;
  }
  if(move) {if(editing) adjust(move);else menuRow=(menuRow+move+10)%10;}
  if(c==shake::KeyEvent::Short) {
    if(menuRow<=6) editing=!editing;
    else if(menuRow==9) {exitMenu=true;exitChoice=0;}
  }
}
void menuText(int row,char* text,size_t n) {
  switch(row) {
    case 0:snprintf(text,n,"打印时长  %u 秒",pending.duration);break;
    case 1:if(pending.range) snprintf(text,n,"固定量程  %.2f mV",shake::ranges[pending.range-1]/1000);else snprintf(text,n,"量程设定  自动");break;
    case 2:snprintf(text,n,"高通截止  %.1f Hz",pending.highpass);break;
    case 3:snprintf(text,n,"低通截止  %.1f Hz",pending.lowpass);break;
    case 4:snprintf(text,n,"显示单位  %s",pending.velocity?"mm/s":"mV");break;
    case 5:snprintf(text,n,"打印图标  %s",pending.logo?"开":"关");break;
    case 6:snprintf(text,n,"震相阈值  %.2f",pending.threshold);break;
    case 7:snprintf(text,n,"手机授时  192.168.4.1");break;
    case 8:snprintf(text,n,"关于 / 帮助");break;
    case 9:snprintf(text,n,"应用或放弃修改");break;
  }
}
void draw() {
  oled.clearBuffer();oled.setFont(u8g2_font_wqy12_t_gb2312);
  char text[96];
  if(menu) {
    if(exitMenu) {
      const char* options[]={"应用退出","放弃修改","继续设置"};
      oled.drawUTF8(0,12,"设置仅本次开机有效");
      for(int i=0;i<3;i++) {snprintf(text,sizeof(text),"%s %s",exitChoice==i?">":" ",options[i]);oled.drawUTF8(0,28+i*15,text);}
    } else if(menuRow==7) {
      oled.drawUTF8(0,12,"手机连接设备 Wi-Fi 后授时");oled.drawStr(0,28,ssid);
      oled.drawStr(0,43,"http://192.168.4.1");oled.drawUTF8(0,61,"左右切换  长按中键返回");
    } else if(menuRow==8) {
      oled.drawStr(0,12,version);oled.drawUTF8(0,28,"科普地震仪 V2  /  LPPNM");
      oled.drawUTF8(0,44,"左/右量程 中键打印 长按菜单");oled.drawUTF8(0,60,"打印电源需确认 9 V / 2 A");
    } else {
      oled.drawUTF8(0,12,editing?"左右调整  中键完成":"左右选择  中键编辑");
      for(int i=0;i<3;i++) {int row=(menuRow+i)%10;menuText(row,text,sizeof(text));oled.drawStr(0,29+i*15,i?" ":">");oled.drawUTF8(12,29+i*15,text);}
    }
  } else {
    static float copy[historySize];static Pick pickCopy[pickCapacity];
    uint32_t count,n;int64_t last,offset;PrintJob j;Settings cfg;
    portENTER_CRITICAL(&mux);
    count=std::min(historyCount,uint32_t(historySize));last=historyLastUs;cfg=settings;j=job;offset=epochOffset;
    for(uint32_t i=0;i<count;i++) copy[i]=history[(historyCount-count+i)%historySize];
    n=std::min(pickCount,uint32_t(pickCapacity));for(uint32_t i=0;i<n;i++) pickCopy[i]=picks[(pickCount-n+i)%pickCapacity];
    portEXIT_CRITICAL(&mux);
    float peak=50;
    for(uint32_t i=count>1250?count-1250:0;i<count;i++) peak=std::max(peak,std::abs(copy[i])*1.1f);
    portENTER_CRITICAL(&mux);autoRange=peak;portEXIT_CRITICAL(&mux);
    float range=j.active?j.range:cfg.range?shake::ranges[cfg.range-1]:peak;
    snprintf(text,sizeof(text),"%s %.3f %s  %.1f-%.0fHz",j.active?"锁定":cfg.range?"固定":"自动",cfg.velocity?range/(1000*cfg.sensitivity):range/1000,cfg.velocity?"mm/s":"mV",cfg.highpass,cfg.lowpass);
    oled.drawUTF8(0,10,text);
    for(int x=0;x<256;x+=8) oled.drawPixel(x,35);
    for(int x=0;x<256;x++) {
      uint32_t a=uint32_t(x)*historySize/256,b=uint32_t(x+1)*historySize/256;
      // Left-padding during startup, with the right edge always representing now.
      int64_t shift=int64_t(historySize)-count;
      int ia=int64_t(a)-shift,ib=int64_t(b)-shift;if(ib<=0||ia>=int(count)) continue;
      ia=std::max(ia,0);ib=std::min(ib,int(count));float lo=INFINITY,hi=-INFINITY;
      for(int k=ia;k<ib;k++) if(std::isfinite(copy[k])) {lo=std::min(lo,copy[k]);hi=std::max(hi,copy[k]);}
      if(!std::isfinite(lo)) continue;
      int y0=shake::amplitude_pixel(hi,range,35,21),y1=shake::amplitude_pixel(lo,range,35,21);
      oled.drawVLine(x,y0,y1-y0+1);
    }
    int64_t leftUs=last-9'996'000;
    oled.setFont(u8g2_font_5x7_tf);
    for(uint32_t i=0;i<n;i++) if(pickCopy[i].time_us>=leftUs&&pickCopy[i].time_us<=last) {
      int x=std::clamp(int((pickCopy[i].time_us-leftUs)*255/9'996'000),0,255);
      for(int y=13;y<=55;y+=3) oled.drawPixel(x,y);
      oled.drawStr(std::min(x+2,249),20,pickCopy[i].phase==1?"P":"S");
    }
    if(!adcReady) snprintf(text,sizeof(text),"ADC ERROR");
    else if(j.active) snprintf(text,sizeof(text),"%s %ds | paper delay 6s",j.dry?"DRY PRINT":"PRINT",int(std::max<int64_t>(0,j.end+6'000'000-esp_timer_get_time())/1000000));
    else {
      auto m=metricsCopy();
      if(!m.coverageStart) snprintf(text,sizeof(text),"AI warmup %lu/3072 | 10s -> now",(unsigned long)std::min<uint32_t>(m.modelSamples,3072));
      else if(n) {char clock[64];clockString(clock,sizeof(clock),pickCopy[n-1].time_us,offset);snprintf(text,sizeof(text),"P%.2f S%.2f %c %s",m.pProbability,m.sProbability,pickCopy[n-1].phase==1?'P':'S',clock);}
      else snprintf(text,sizeof(text),"P:%.3f S:%.3f threshold:%.2f",m.pProbability,m.sProbability,cfg.threshold);
    }
    oled.drawStr(0,64,text);
  }
  if(int32_t(noticeUntil-millis())>0) {oled.setDrawColor(0);oled.drawBox(0,48,256,16);oled.setDrawColor(1);oled.setFont(u8g2_font_wqy12_t_gb2312);oled.drawUTF8(0,62,notice);}
  flushDisplay();
}

void setupWeb() {
  snprintf(ssid,sizeof(ssid),"USTCShake-Time-%04X",unsigned(ESP.getEfuseMac()&0xffff));
  WiFi.persistent(false);WiFi.mode(WIFI_AP);WiFi.softAP(ssid);
  web.on("/",HTTP_GET,[]{web.send(200,"text/html; charset=utf-8",R"HTML(<!doctype html><meta name="viewport" content="width=device-width"><title>科普地震仪授时</title><style>body{font:18px system-ui;margin:40px auto;max-width:500px;padding:20px}button{font:inherit;padding:15px}</style><h1>科普地震仪 V2</h1><p>用本机时间校准地震仪。震相记录以 UTC 打印，未校准时显示开机相对时间。</p><button onclick="fetch('/time',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'unix_ms='+Date.now()}).then(r=>r.text()).then(t=>document.querySelector('#result').textContent=t)">校准时间</button><p id="result"></p><p>LPPNM：3072 点，每 128 点更新。模型预热约 31 秒。</p>)HTML");});
  web.on("/time",HTTP_POST,[]{int64_t ms=strtoll(web.arg("unix_ms").c_str(),nullptr,10);if(ms<1577836800000LL||ms>4102444800000LL){web.send(400,"text/plain; charset=utf-8","时间无效");return;}setEpoch(ms);web.send(200,"text/plain; charset=utf-8","校时完成");});
  web.begin();
}
void setup() {
  Serial.setTxBufferSize(4096);Serial.setTxTimeoutMs(0);Serial.begin(115200);
  for(int pin:{board::key_left,board::key_center,board::key_right}) pinMode(pin,INPUT);
  initializeDisplay();oled.drawUTF8(0,24,"科普地震仪 V2");oled.drawStr(0,44,"LPPNM INT8 / 3072 / 128");flushDisplay();
  Wire.begin(board::sda,board::scl,100000);Wire.setTimeOut(2);
  bool ok=writeAdc(3,0x8000)&&writeAdc(2,0)&&writeAdc(1,0x82a0);
  int16_t config=0;ok=ok&&readAdc(1,config)&&((uint16_t(config)&0x7fff)==0x02a0);
  adcReady=ok;
  if(!ok) {message("ADC 配置失败");draw();while(true) {Serial.println("ADC CONFIG ERROR; firmware not sampling");delay(1000);}}
  printer.begin(9600,SERIAL_8N1,-1,board::printer_tx);delay(500);
  printer.write(board::printer_setup,sizeof(board::printer_setup));printer.flush();delay(100);printer.updateBaudRate(board::printer_baud);
  inferenceQueue=xQueueCreate(512,sizeof(Sample));usbQueue=xQueueCreate(128,sizeof(Sample));paperQueue=xQueueCreate(1024,sizeof(PaperRow));
  picker=new lppnm::OnlinePicker(lppnm_model::nodes,lppnm_model::count,lppnm_model::class_head,lppnm_model::time_head);
  if(!inferenceQueue||!usbQueue||!paperQueue||!picker) {message("内存不足");draw();while(true) delay(1000);}
  setupWeb();
  BaseType_t a=xTaskCreate(acquire,"adc250",4096,nullptr,20,&acquisitionHandle);
  BaseType_t b=xTaskCreate(paperTask,"paper",8192,nullptr,3,&paperHandle);
  BaseType_t c=xTaskCreate(infer,"lppnm",6144,nullptr,2,&inferenceHandle);
  if(a!=pdPASS||b!=pdPASS||c!=pdPASS) {message("任务创建失败");draw();while(true) delay(1000);}
  esp_timer_create_args_t args{};args.callback=tick;args.name="sample250";args.skip_unhandled_events=true;
  ESP_ERROR_CHECK(esp_timer_create(&args,&sampleTimer));samplingOrigin=esp_timer_get_time();
  ESP_ERROR_CHECK(esp_timer_start_periodic(sampleTimer,4000));hello();
}
void loop() {
  static uint32_t lastDraw=0;
  buttons();serialPoll();web.handleClient();
  if(millis()-lastDraw>=100) {lastDraw=millis();draw();}
  delay(1);
}
