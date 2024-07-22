# Seamless(S2T) Decoder ONNX模型导出
[Seamless官方仓库](https://github.com/facebookresearch/seamless_communication)未提供onnx导出脚本，本导出教程基于官方代码进行修改，导出的encoder onnx模型精度与官方pt模型一致。

## 1. 特性
* 支持离线M4t(S2T任务)的encoder onnx模型导出

## 2. 代码修改说明
主要修改内容包括：
* 增加onnx导出代码
* 移除forward自定义输入参数类型，替换为PyTorch导出onnx支持的输入类型
* 变更KV cache作为输入输出
* 为了确保固定序列长度输入的有效性，增加mask作为输入
* 平替一些算子，例如将隐式广播替换为了具体PyTorch函数
* 重设mask的填充值
* 代码不会进行推理，导出完成即退出

## 3. 安装和配置依赖环境
encoder的onnx模型导出需要在x86主机上完成。
为了导出encoder模块，需要进行如下的环境准备
```bash
git clone https://github.com/TangLinJie/seamless_communication_encoder_export_onnx.git
cd seamless_communication_encoder_export_onnx
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
git clone https://github.com/TangLinJie/fairseq2_encoder_export_onnx.git
cd fairseq2_encoder_export_onnx/
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchaudio==2.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
# 若使用conda需要安装
conda install -c conda-forge libsndfile==1.0.31
```

## 4. M4t(S2T任务) Encoder ONNX模型导出
- step1：下载pt模型
在导出时，代码会自动下载pt模型，值得注意的是对于huggingface上的模型下载很可能出现网络问题，可下载好以后，修改`src/seamless_communication/cards/seamlessM4T_v2_large.yaml`和`src/seamless_communication/cards/vocoder_v2.yaml`中的模型路径。

- step2：导出onnx模型
执行如下命令导出encoder的所有子模块(包括模块encoder frontend、encoder)
```bash
cd ..
# 创建结果文件夹
mkdir m4t_unity_speech_encoder_frontend m4t_unity_speech_encoder
python export_m4t_offline.py
```