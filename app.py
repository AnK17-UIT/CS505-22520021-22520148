import gradio as gr
import torch
import os
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification
)
from peft import PeftModel

# Qwen Config
QWEN_BASE_NAME = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_ADAPTER_PATH = "./results/Qwen-Final-Unified-NLI"

# PhoBERT Config
PHOBERT_PATH = "./results/phobert-large-hallu-finetuned"

print("⏳ Đang khởi động hệ thống trên CPU...")
# Load model
print("--- Loading Qwen Tokenizer ---")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE_NAME, trust_remote_code=True)
if qwen_tokenizer.pad_token is None:
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
qwen_tokenizer.padding_side = "left"

print("--- Loading Qwen Base Model ---")
try:
    qwen_base = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE_NAME,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
except Exception as e:
    print(f"❌ Lỗi load Qwen Base: {e}")
    exit()

if os.path.exists(QWEN_ADAPTER_PATH):
    print("--- Loading Qwen Adapter ---")
    qwen_model = PeftModel.from_pretrained(qwen_base, QWEN_ADAPTER_PATH)
    qwen_model.eval()
    print("✅ Qwen Loaded Successfully!")
else:
    print(f"❌ Không tìm thấy Adapter tại: {QWEN_ADAPTER_PATH}")
    exit()

print("--- Loading PhoBERT-Large ---")
if os.path.exists(PHOBERT_PATH):
    try:
        phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_PATH)
        phobert_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_PATH)
        phobert_model.to("cpu")
        phobert_model.eval()
        print("✅ PhoBERT Loaded Successfully!")
    except Exception as e:
        print(f"❌ Lỗi load PhoBERT: {e}")
        phobert_model = None
else:
    print(f"⚠️ Cảnh báo: Không tìm thấy folder PhoBERT tại {PHOBERT_PATH}. Chế độ PhoBERT sẽ bị tắt.")
    phobert_model = None


# --- Map nhãn cho Qwen (Sinh văn bản) ---
def map_label_qwen(raw_output):
    raw_lower = raw_output.lower().strip()
    if "entailment" in raw_lower:
        return "✅ Entailment (Tin cậy)"
    elif "contradiction" in raw_lower:
        return "❌ Intrinsic-Hal (Mâu thuẫn)"
    elif "neutral" in raw_lower:
        return "⚠️ Extrinsic-Hal (Bịa đặt)"
    else:
        return f"❓ Unknown ({raw_output})"

# --- Map nhãn cho PhoBERT (Phân loại) ---
# Giả định thứ tự nhãn lúc train PhoBERT là: 0: Entailment, 1: Intrinsic, 2: Extrinsic
# Nếu bạn train khác thứ tự, hãy sửa lại dict này
phobert_id2label = {
    0: "✅ Entailment (Tin cậy)",
    1: "❌ Intrinsic-Hal (Mâu thuẫn)",
    2: "⚠️ Extrinsic-Hal (Bịa đặt)"
}

def format_prompt_qwen(context, statement, domain):
    if domain == "Y tế (ViMedNLI)":
        role = "You are a medical AI assistant."
        note = ""
    else:
        role = "You are an AI expert in Vietnamese Natural Language Inference (NLI)."
        note = "Note: The input text covers various domains and may contain complex, tricky phrasing or subtle logical traps. Analyze carefully."

    return f"""{role} Your task is to determine the logical relationship between the Context and the Statement.
{note}
Context: {context}
Statement: {statement}

Based on the context, classify the statement as one of the following:
- entailment
- neutral
- contradiction

Answer:
"""

def predict_all(context, statement, domain):
    if not context or not statement:
        return "⚠️ Trống", "⚠️ Trống", "⚠️ Trống"

    # --- 1. DỰ ĐOÁN VỚI QWEN (Fine-tuned & Base) ---
    full_prompt = format_prompt_qwen(context, statement, domain)
    messages = [{"role": "user", "content": full_prompt}]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer([text], return_tensors="pt").to("cpu")

    # Qwen Fine-tuned
    qwen_model.enable_adapter_layers()
    with torch.no_grad():
        out_ft = qwen_model.generate(**inputs, max_new_tokens=30, pad_token_id=qwen_tokenizer.eos_token_id, temperature=0.1, do_sample=False)
    res_ft = map_label_qwen(qwen_tokenizer.decode(out_ft[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())

    # Qwen Base
    with qwen_model.disable_adapter():
        with torch.no_grad():
            out_base = qwen_model.generate(**inputs, max_new_tokens=30, pad_token_id=qwen_tokenizer.eos_token_id, temperature=0.1, do_sample=False)
    res_base = map_label_qwen(qwen_tokenizer.decode(out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())

    # --- 2. DỰ ĐOÁN VỚI PHOBERT ---
    if phobert_model:
        # PhoBERT nối câu bằng token đặc biệt (<s> sentence1 </s> </s> sentence2 </s>)
        # Tokenizer của PhoBERT tự xử lý việc này khi truyền 2 câu
        phobert_inputs = phobert_tokenizer(
            context, 
            statement, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256,
            padding=True
        ).to("cpu")
        
        with torch.no_grad():
            logits = phobert_model(**phobert_inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
            
        label_text = phobert_id2label.get(pred_idx, "Unknown")
        res_phobert = f"{label_text}\n(Độ tin cậy: {confidence:.2%})"
    else:
        res_phobert = "⚠️ Model not loaded"

    return res_ft, res_base, res_phobert

custom_css = """
.output-box textarea { 
    font-size: 18px !important; 
    font-weight: bold !important; 
}
"""

with gr.Blocks(title="Hallucination Detection System", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🕵️ Hệ thống Phát hiện Hallucination (Multi-Model)")
    gr.Markdown("So sánh kết quả giữa **Qwen-4B (LLM)** và **PhoBERT-Large (Encoder)** trên CPU.")
    
    with gr.Row():
        # Cột Input
        with gr.Column(scale=1):
            inp_domain = gr.Dropdown(
                ["Y tế (ViMedNLI)", "Đa lĩnh vực"], 
                value="Đa lĩnh vực", label="Lĩnh vực (Domain)"
            )
            inp_context = gr.Textbox(lines=6, placeholder="Nhập ngữ cảnh...", label="Context")
            inp_statement = gr.Textbox(lines=3, placeholder="Nhập nhận định...", label="Statement")
            
            with gr.Row():
                btn_run = gr.Button("🚀 Phân tích", variant="primary")
                gr.ClearButton([inp_context, inp_statement])

        # Cột Output
        with gr.Column(scale=1):
            gr.Markdown("### 🏆 Kết quả Phân tích")
            
            # Group 1: Qwen
            with gr.Group():
                gr.Markdown("#### 🤖 Qwen-4B (Fine-tuned w/ QLoRA)")
                out_ft = gr.Textbox(label="Kết quả", elem_classes="output-box")
            
            # Group 2: PhoBERT
            with gr.Group():
                gr.Markdown("#### 🦅 PhoBERT-Large (Fine-tuned)")
                out_phobert = gr.Textbox(label="Kết quả", elem_classes="output-box")
            
            gr.Markdown("---")
            
            # Group 3: Base Model (Tham chiếu)
            with gr.Group():
                gr.Markdown("#### 👶 Qwen-4B Base (Gốc)")
                out_base = gr.Textbox(label="Kết quả", elem_classes="output-box")

    btn_run.click(
        predict_all, 
        [inp_context, inp_statement, inp_domain], 
        [out_ft, out_base, out_phobert]
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False)
