import os
import torch
from typing import Iterable
import gradio as gr
from transformers import (
    AutoProcessor,
    HunYuanVLForConditionalGeneration, 
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# Import spaces if available, otherwise mock it
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

# --- Theme Definition ---
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()
css = """
#main-title h1 { font-size: 2.3em !important; }
#output-title h2 { font-size: 2.1em !important; }
"""

# --- Model Loading ---

# HunyuanOCR
MODEL_HUNYUAN = "tencent/HunyuanOCR"
print(f"Loading {MODEL_HUNYUAN}...")
try:
    processor_hy = AutoProcessor.from_pretrained(MODEL_HUNYUAN, use_fast=False)
    model_hy = HunYuanVLForConditionalGeneration.from_pretrained(
        MODEL_HUNYUAN,
        attn_implementation="eager", 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    ).eval()
    print("✅ HunyuanOCR loaded successfully.")
except Exception as e:
    print(f"❌ Error loading HunyuanOCR: {e}")
    raise e

# --- Helper Functions ---

def clean_repeated_substrings(text):
    """Clean repeated substrings in text (Specific fix for Hunyuan output issues)"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]  
    return text

# --- Main Inference Logic ---

@spaces.GPU
def run_hunyuan_model(
    image, 
    custom_prompt,
    max_new_tokens
):
    if image is None:
        return "Please upload an image."

    # Default prompt if empty
    query = custom_prompt if custom_prompt else "检测并识别图片中的文字，将文本坐标格式化输出。"
    
    # Hunyuan template structure
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": query},
            ],
        }
    ]
    
    try:
        # Note: Hunyuan processor expects specific handling
        texts = [processor_hy.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        inputs = processor_hy(text=texts, images=image, padding=True, return_tensors="pt")
        inputs = inputs.to(model_hy.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model_hy.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False 
            )
            
        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        output_text = processor_hy.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        final_text = clean_repeated_substrings(output_text)
        return final_text

    except Exception as e:
        return f"Error during generation: {str(e)}"

# --- Gradio UI ---

image_examples = [
    ["examples/1.jpg"],
    ["examples/2.jpg"],
    ["examples/3.jpg"],
]

with gr.Blocks() as demo:
    gr.Markdown("# **HunyuanOCR Demo**", elem_id="main-title")
    gr.Markdown("Demonstration of **tencent/HunyuanOCR** for text extraction and recognition.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"], height=350)
            
            custom_prompt = gr.Textbox(
                label="Custom Query / Prompt", 
                placeholder="Extract text...", 
                lines=2, 
                value="检测并识别图片中的文字，将文本坐标格式化输出。"
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_new_tokens = gr.Slider(minimum=128, maximum=8192, value=2048, step=128, label="Max New Tokens")
            
            submit_btn = gr.Button("Perform OCR", variant="primary")
            
            if os.path.exists("examples"):
                gr.Examples(examples=image_examples, inputs=image_input)

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Recognized Text", lines=20, interactive=True)

    submit_btn.click(
        fn=run_hunyuan_model,
        inputs=[image_input, custom_prompt, max_new_tokens],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(css=css, theme=steel_blue_theme, ssr_mode=False, show_error=True, share=True)
