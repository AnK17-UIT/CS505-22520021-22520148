import gradio as gr
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Cấu hình CPU
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "./results/Qwen-Final-Unified-NLI" 

print("Khởi động trên CPU")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load Base Model
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="cpu", 
        trust_remote_code=True,
        torch_dtype=torch.float32 
    )
except Exception as e:
    print(f"Lỗi load model: {e}")
    exit()

# Load Adapter
if os.path.exists(ADAPTER_PATH):
    print("Đang gắn Adapter vào Base Model...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Load Adapter thành công!")
else:
    print(f"Không tìm thấy Adapter tại: {ADAPTER_PATH}")
    exit()

# --- HÀM HẬU XỬ LÝ NHÃN (MỚI THÊM) ---
def map_label(raw_output):
    """
    Chuyển đổi nhãn NLI sang nhãn Hallucination Detection
    """
    raw_lower = raw_output.lower().strip()
    
    if "entailment" in raw_lower:
        return "✅ Entailment (Tin cậy)"
    elif "contradiction" in raw_lower:
        return "❌ Intrinsic-Hal (Mâu thuẫn)"
    elif "neutral" in raw_lower:
        return "⚠️ Extrinsic-Hal (Bịa đặt/Không kiểm chứng)"
    else:
        # Trường hợp model trả lời linh tinh hoặc đang suy nghĩ (thinking process)
        return f"❓ Unknown ({raw_output})"

# Predict
def format_prompt(context, statement, domain):
    if domain == "Y tế (ViMedNLI)":
        role = "You are a medical AI assistant."
        note = ""
    else:
        role = "You are an AI expert in Vietnamese Natural Language Inference (NLI)."
        note = "Note: The input text covers various domains and may contain complex, tricky phrasing or subtle logical traps. Analyze carefully."

    prompt = f"""{role} Your task is to determine the logical relationship between the Context and the Statement.
{note}
Context: {context}
Statement: {statement}

Based on the context, classify the statement as one of the following:
- entailment
- neutral
- contradiction

Answer:
"""
    return prompt

def predict_comparison(context, statement, domain):
    if not context or not statement:
        return "⚠️ Chưa nhập dữ liệu", "⚠️ Chưa nhập dữ liệu"

    full_prompt = format_prompt(context, statement, domain)
    
    messages = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    # 1. Model Fine-tuned
    model.enable_adapter_layers()
    with torch.no_grad():
        outputs_ft = model.generate(
            **inputs, 
            max_new_tokens=30, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1, 
            do_sample=False
        )
    raw_ft = tokenizer.decode(outputs_ft[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    result_ft = map_label(raw_ft) # <--- Áp dụng hàm map nhãn

    # 2. Model Base
    with model.disable_adapter():
        with torch.no_grad():
            outputs_base = model.generate(
                **inputs, 
                max_new_tokens=30, 
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1, 
                do_sample=False
            )
    raw_base = tokenizer.decode(outputs_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    result_base = map_label(raw_base) # <--- Áp dụng hàm map nhãn

    return result_ft, result_base

# Giao diện
custom_css = """
.output-box textarea { 
    font-size: 20px !important; 
    font-weight: bold !important; 
}
"""

with gr.Blocks(title="NLI Local Demo (CPU)") as demo:
    gr.Markdown("# 🕵️ Hệ thống Phát hiện Hallucination (Local Demo)")
    gr.Markdown("Chạy trên CPU - So sánh giữa Base Model và Fine-tuned Model (QLoRA)")
    
    with gr.Row():
        with gr.Column():
            inp_domain = gr.Dropdown(
                ["Y tế (ViMedNLI)", "Đa lĩnh vực"], 
                value="Đa lĩnh vực", label="Domain"
            )
            inp_context = gr.Textbox(lines=5, placeholder="Nhập ngữ cảnh (Context)...", label="Context")
            inp_statement = gr.Textbox(lines=2, placeholder="Nhập nhận định (Statement)...", label="Statement")
            
            with gr.Row():
                btn_run = gr.Button("🚀 Chạy Dự Đoán", variant="primary")
                btn_clear = gr.ClearButton([inp_context, inp_statement])

        with gr.Column():
            gr.Markdown("### 📊 Kết quả Phân tích")
            out_ft = gr.Textbox(label="Fine-tuned Model (Đề xuất)", elem_classes="output-box")
            out_base = gr.Textbox(label="Base Model (Gốc)", elem_classes="output-box")

    btn_run.click(predict_comparison, [inp_context, inp_statement, inp_domain], [out_ft, out_base])

if __name__ == "__main__":
    demo.launch(
        server_name="localhost", 
        server_port=7860, 
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css
    )