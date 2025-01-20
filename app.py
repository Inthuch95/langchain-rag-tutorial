import openai
import os
import gradio as gr

from dotenv import load_dotenv

# Replace 'your-openai-api-key' with your actual OpenAI API key
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def chat_with_openai(prompt, model="gpt-4o-mini"):
    """
    Sends the user's prompt to the OpenAI API and returns the response.
    
    Args:
        prompt (str): The user's input message.
        model (str): The OpenAI model to use (default is "gpt-4o-mini").

    Returns:
        str: The response from the OpenAI API.
    """
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with OpenAI LLM")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your Prompt", placeholder="Type your message here...", lines=4)
            model_choice = gr.Radio(
                choices=["gpt-4o-mini", "gpt-3.5-turbo"],
                value="gpt-4o-mini",
                label="Model"
            )
            submit_button = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Response", placeholder="The model's response will appear here...", lines=10)

    submit_button.click(chat_with_openai, inputs=[user_input, model_choice], outputs=output)

# Run the Gradio app
demo.launch()
