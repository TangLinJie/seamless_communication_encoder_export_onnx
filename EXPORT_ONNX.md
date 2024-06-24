## 依赖
需要安装https://github.com/TangLinJie/fairseq2_encoder_export_onnx.git，其他库未修改
## 导出SeamlessStreaming模型
导出这类模型需要安装本库，安装完以后执行类似如下命令：
```bash
streaming_evaluate --task s2tt --data-file ./cvssc_ja/test.tsv --audio-root-dir ./cvssc_ja/test --output ./test --tgt-lang eng --dtype fp32 --device cpu
```
其中test.tsv文件是tsv格式，需要有列名，类似如下。./cvssc_ja/test目录下包含了相应文件名的语音文件。
```
audio	tgt_text
common_voice_ja_19482479.mp3.wav	put some salt in the boiling water
common_voice_ja_19482480.mp3.wav	kimura showed me his photos
common_voice_ja_19482481.mp3.wav	she has her arm in a cast
common_voice_ja_19482487.mp3.wav	father is tallest of all of us
common_voice_ja_19482488.mp3.wav	she ran towards me her hair rippling in the wind
common_voice_ja_19482489.mp3.wav	my dad wants me to become an engineer
common_voice_ja_19482490.mp3.wav	please take the injured to the hospital
```
## 导出m4t模型
导出这类模型需要安装本库，安装完以后执行类似如下命令：
```bash
python3 export_m4t_offline.py
```
