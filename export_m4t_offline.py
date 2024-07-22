import torch
import os
from seamless_communication.inference import Translator
import time

translator = Translator(
    "seamlessM4T_v2_large",
    "vocoder_v2",
    device=torch.device("cpu"), # works if I use device="cpu" + dtype=torch.float16
    dtype=torch.float32,
)
file_dir = "test_data"
filepaths = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir)]
total_time = 0
for filepath in filepaths:
    start_t = time.time()
    text_output, _ = translator.predict(
        input=filepath,
        src_lang="cmn", # or, "jpn"
        task_str="s2tt",
        tgt_lang="cmn",
    )
    total_time += time.time() - start_t
    with open('./results/' + filepath.split('/')[-1].split('.')[0]+'.txt', 'w') as fw:
        fw.write(str(text_output[0]))
    print(f"Translated text to English: {text_output[0]}")
print(total_time)
print(total_time / len(filepaths))
