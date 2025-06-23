from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
ckpt_path = "/data/phd/jinjiachun/ckpt/Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    ckpt_path, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(ckpt_path)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_parameters / 1e9:.2f}B")