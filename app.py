import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# 1. Load Model & Tokenizer
# Replace with your Hugging Face model repository ID
MODEL_ID = "Pranjan007/roberta-mcq-solver"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_ID)
model.eval()

# 2. MCQ Inference Logic (CPU Execution)
def predict_mcq(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not prompt.strip():
        return "Please enter a valid question prompt."
    
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    option_labels = ['A', 'B', 'C', 'D', 'E']
    
    # Format input pairs: "Question: ... Option X: ..."
    first_sentences = [f"Question: {prompt}"] * 5
    second_sentences = [f"Option {label}: {text}" for label, text in zip(option_labels, options)]
    
    # Tokenize input pairs
    inputs = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt"
    )
    
    # Reshape input tensors to (1, 5, sequence_length)
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (1, 5)
        probabilities = F.softmax(logits, dim=1).squeeze(0).numpy()
        
    # Sort choices by confidence score
    option_probs = list(zip(option_labels, options, probabilities))
    option_probs.sort(key=lambda x: x[2], reverse=True)
    
    top3_string = " ".join([item[0] for item in option_probs[:3]])
    
    # Generate Formatted Output
    output_md = f"### 🏆 Top-3 Predicted Ranking: `{top3_string}`\n\n"
    output_md += "| Rank | Choice | Option Text | Confidence Probability |\n"
    output_md += "| :--- | :---: | :--- | :--- |\n"
    
    for rank, (label, text, prob) in enumerate(option_probs, 1):
        output_md += f"| **#{rank}** | **Option {label}** | {text} | **{prob * 100:.2f}%** |\n"
        
    return output_md

# 3. Gradio Interface Definition
with gr.Blocks(title="Smart MCQ Solver - RoBERTa-base", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Smart MCQ Solver (RoBERTa-base)")
    gr.Markdown("Enter a question prompt along with 5 multiple-choice options to view the **Top-3 ranking** and **probabilities**.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Question Prompt", lines=3)
            opt_a = gr.Textbox(label="Option A")
            opt_b = gr.Textbox(label="Option B")
            opt_c = gr.Textbox(label="Option C")
            opt_d = gr.Textbox(label="Option D")
            opt_e = gr.Textbox(label="Option E")
            submit_btn = gr.Button("Predict Top-3 Choices", variant="primary")
            
        with gr.Column():
            result_output = gr.Markdown(label="Prediction Results")

    submit_btn.click(
        fn=predict_mcq, 
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e], 
        outputs=result_output
    )

# Important for Docker: Bind to 0.0.0.0 and port 7860
demo.launch(server_name="0.0.0.0", server_port=7860)