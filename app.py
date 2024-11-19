from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set the padding token to the eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize the Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        response = get_Chat_response(msg)
        return jsonify(response)
    except Exception as e:
        return jsonify(str(e))

def get_Chat_response(text):
    chat_history_ids = None
    new_user_input_ids = tokenizer(
        str(text) + tokenizer.eos_token,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    for step in range(5):
        if chat_history_ids is None:
            bot_input_ids = new_user_input_ids['input_ids']
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids['input_ids']], dim=-1)
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=new_user_input_ids['attention_mask']
        )
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
