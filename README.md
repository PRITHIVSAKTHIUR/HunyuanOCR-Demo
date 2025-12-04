# **HunyuanOCR-Demo**

A Gradio-based demonstration application for the Tencent HunyuanOCR model, focused on optical character recognition (OCR) tasks such as text detection, extraction, and coordinate formatting from images. Users can upload images, customize prompts (e.g., for Chinese/English text), and generate structured outputs with advanced generation controls.

## Features

- **Image Upload and Processing**: Supports direct upload or clipboard paste; processes images via PIL for text recognition.
- **Custom Prompts**: Tailor queries like "检测并识别图片中的文字，将文本坐标格式化输出。" (Detect and recognize text in the image, format text coordinates output) for precise extraction.
- **Advanced Generation Controls**: Adjustable max new tokens (up to 8192) for handling complex documents.
- **Output Handling**: Cleaned text to remove repetitions; interactive textbox with copy button for easy use.
- **Custom Theme**: SteelBlueTheme with gradients and enhanced typography for a professional interface.
- **Examples Integration**: Built-in sample images for quick testing (e.g., documents, receipts).
- **Queueing Support**: Handles up to 10 concurrent inferences for smooth multi-user access.
- **Error Resilience**: Graceful handling of loading and generation errors with informative messages.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16; falls back to CPU with float32).
- Git for cloning submodules.
- Hugging Face account (optional, for model caching via `huggingface_hub`).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/HunyuanOCR-Demo.git
   cd HunyuanOCR-Demo
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
   git+https://github.com/huggingface/accelerate.git
   git+https://github.com/huggingface/diffusers.git
   git+https://github.com/huggingface/peft.git
   huggingface_hub
   qwen-vl-utils
   sentencepiece
   opencv-python
   torchvision
   supervision
   matplotlib
   easydict
   kernels
   gradio
   einops
   spaces
   addict
   hf_xet
   torch
   numpy
   av
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Upload Image**: Drag-and-drop or paste an image (e.g., scanned document, sign, or multilingual text).

2. **Set Prompt**: Enter a custom query in the textbox. Default: "检测并识别图片中的文字，将文本坐标格式化输出。" for formatted text with coordinates.

3. **Configure Settings**:
   - Expand "Advanced Settings" to adjust max new tokens for longer outputs.

4. **Run Inference**: Click "Perform OCR" to process. Results appear in the output textbox.

5. **View Results**:
   - Text: Structured OCR output (e.g., detected text with bounding box coordinates).
   - Copy or edit the interactive output as needed.

### Example Workflow
- Upload a Chinese receipt image.
- Use default prompt for coordinate-formatted extraction.
- Set max new tokens to 2048 for detailed results.
- Output: List of text segments with positions like "Text: '价格', Coordinates: [x1, y1, x2, y2]".

## Troubleshooting

- **Model Loading Errors**: Verify CUDA setup; check console for `torch.version.cuda`. Use `attn_implementation="eager"` to avoid SDPA issues.
- **Out of Memory**: Reduce max new tokens or use CPU fallback; monitor with `nvidia-smi`.
- **Import Issues**: Install `spaces` only for Hugging Face Spaces deployment; it's mocked locally.
- **Repeated Output**: Automatically cleaned via `clean_repeated_substrings`; increase threshold if needed.
- **Generation Fails**: Ensure prompt is valid; test with default for baseline.
- **UI Launch**: If `ssr_mode=False` causes issues, set to `True` for server-side rendering.

## Contributing

Contributions are encouraged! Open issues for bugs or enhancements (e.g., batch processing, additional post-processing). Fork, create a branch, and submit a pull request with tests. Potential areas:
- Integration with other OCR models.
- Export options (e.g., JSON coordinates).
- Multilingual prompt templates.

Repository: [https://github.com/PRITHIVSAKTHIUR/HunyuanOCR-Demo.git](https://github.com/PRITHIVSAKTHIUR/HunyuanOCR-Demo.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
