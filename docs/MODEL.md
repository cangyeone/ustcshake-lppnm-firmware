# 增量模型约定

运行时：`model/runtime.hpp`；量化图和权重：`model/artifacts/model.hpp`。
原模型检查点为 pnsn 的 `ckpt/china.lppnm.pt`，模型结构为 `models/LPPNM.py`。

输入 100 Hz；每次 128 点；单路复制为三路；3072 点因果均值与最大绝对偏差归一化。旧样本不会被重新归一化。外部 ADC 的 250 Hz 输入先经过 161 tap Kaiser FIR 的 2/5 多相重采样，补偿 160 ms 群时延。

50 个图节点、39 个融合卷积，全部去除 BatchNorm。卷积为 INT8 权重 / INT8 激活 / INT32 偏置和累加，Q24 INT64 乘数重定标，half-away-from-zero 舍入至 [-127,127]。softmax 和时间回归 sigmoid 使用浮点。
权重逐输出通道量化，激活逐张量静态量化。模型保留完整七类 softmax，类型 1、2 用作 P/S 候选。

每次更新约 631,168 MAC，不重算整个 3072 点窗口；整窗相应计算量约 15,148,032 MAC。
分层缓存约 43,874 bytes，另有因果归一化历史、索引队列和图对象。运行时只在构造阶段分配，各次 push/reset 不分配堆内存。

网络双侧上下文约需 212–268 个未来样本。128 点调度使输出相对时间格点的等待为 2.48–3.68 秒，另加 FIR、板端实际推理和候选确认耗时。超过阈值的局部峰经 5 个下降格点（约 0.4 秒）确认；同类型一秒内去重，P/S 独立。

严格区分：浮点融合 ONNX 是参考模型，不是板上运行的流式 INT8 模型。板上不依赖 Torch、ONNX Runtime 或 ESP-DL。

`validation.json` 包含融合一致性、整段/流式整数逐位一致性、校准区间与独立留出区间误差。
`native_validation.json` 包含 C++ / Python 逐位一致性、归一化边界与舍入测试。
这些结果并非设备单分量实际 P/S 召回率或 F1；0.01 阈值的科学精度尚待标注数据评测。

重新生成后可运行：

```sh
c++ -O2 -std=c++17 model/cli.cpp -o model/artifacts/stream_cli
python model/check_native.py
```

`build.py` 会生成完整浮点融合参考、校准图、验证输入及报告。大型中间文件默认被 Git 忽略。生成器不包含上游原始数据或检查点副本。
