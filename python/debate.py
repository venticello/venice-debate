import re
import gradio as gr
import openai
import time
import textwrap
from typing import Generator, List, Dict, Any
from dataclasses import dataclass

from pydantic import SecretStr
from pydantic_settings import BaseSettings

# Constants
DEFAULT_PROTAGONIST_MODEL = "venice-uncensored"
DEFAULT_ANTAGONIST_MODEL = "llama-3.1-405b"
DEFAULT_MAX_TOKENS = 700
DEFAULT_DEBATE_TURNS = 3
TEMPERATURE = 0.6
THINKING_ANIMATION = ["Thinking.", "Thinking..", "Thinking..."]

MODELS = [
    "venice-uncensored",
    "qwen3-235b",
    "deepseek-r1-671b",
    "llama-3.1-405b",
    "llama-3.3-70b",
    "dolphin-2.9.2-qwen2-72b",
    "qwen-2.5-qwq-32b",
    "mistral-31-24b",
]

DEFAULT_PROTAGONIST_SYS_PROMPT = (
    "You are a precise and analytical AI debater representing the Protagonist perspective. "
    "Engage directly with your opponent's arguments, referencing specific points they've made throughout the debate where relevant. "
    "Maintain a logical flow and build upon your previous arguments. Your goal is a constructive exchange of ideas."
)

DEFAULT_ANTAGONIST_SYS_PROMPT = (
    "You are a creative and insightful AI debater representing the Antagonist perspective. "
    "Challenge your opponent's points thoughtfully and connect your arguments back to the core topic. "
    "Feel free to refer to earlier statements in the debate to highlight consistencies or contradictions. Aim for a compelling and engaging discussion."
)


@dataclass
class DebateParticipant:
    model_name: str
    system_prompt: str
    is_protagonist: bool

    @property
    def display_name(self) -> str:
        role = "Protagonist" if self.is_protagonist else "Antagonist"
        return f"{role} ({self.model_name})"


class Settings(BaseSettings):
    venice_key: SecretStr

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


def get_venice_client(api_key: str) -> openai.OpenAI:
    """Initialize and validate Venice API client."""
    if not api_key:
        raise gr.Error("Venice API Key is missing!")
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.venice.ai/api/v1")
        client.models.list()
        return client
    except openai.AuthenticationError:
        raise gr.Error("Invalid Venice API Key.")
    except Exception as e:
        raise gr.Error(f"Venice Client Error: {e}")


def validate_inputs(topic: str, venice_key: str, max_tokens: int) -> None:
    """Validate all input parameters."""
    if not topic:
        raise gr.Error("Please provide a debate topic!")
    if not venice_key:
        raise gr.Error("Please provide your Venice API Key!")
    try:
        if max_tokens <= 0:
            raise ValueError("Max tokens must be positive.")
    except (ValueError, TypeError):
        raise gr.Error("Invalid Max Tokens value. Please use the slider or enter a positive number.")


def create_debate_header(topic: str, protagonist: DebateParticipant, antagonist: DebateParticipant,
                         max_tokens: int) -> str:
    """Create the initial debate transcript header."""
    return (
        f"## Debate Topic: {topic}\n\n"
        f"**Settings:**\n"
        f"- Protagonist Model: `{protagonist.model_name}`\n"
        f"- Antagonist Model: `{antagonist.model_name}`\n"
        f"- Max Tokens: {max_tokens}\n"
        f"- Protagonist Persona: *{textwrap.shorten(protagonist.system_prompt, 128)}*\n"
        f"- Antagonist Persona: *{textwrap.shorten(antagonist.system_prompt, 128)}*\n\n"
        f"---\n\n"
    )


def process_streaming_response(
        stream: Any,
        debate_transcript: str,
        status_message: str
) -> Generator[tuple[str, str], None, tuple[str, str]]:
    """Process streaming response and update transcript."""
    full_response = ""
    in_think_block = False

    for chunk in stream:
        if not hasattr(chunk, 'choices') or not chunk.choices:
            continue
        if not hasattr(chunk.choices[0], 'delta') or not hasattr(chunk.choices[0].delta, 'content'):
            continue

        content = chunk.choices[0].delta.content
        if content is None:
            continue

        full_response += content

        # Handle think blocks
        if "<think>" in content:
            in_think_block = True
            debate_transcript += "<blockquote>"
            content = content.replace("<think>", "")
        elif "</think>" in content:
            in_think_block = False
            content = content.replace("</think>", "")
            debate_transcript += content + "</blockquote>"
            continue

        debate_transcript += content
        yield debate_transcript, status_message

    return full_response, debate_transcript


def run_debate(
        topic: str,
        venice_key: str,
        protagonist_model_name: str,
        antagonist_model_name: str,
        protagonist_system_prompt: str,
        antagonist_system_prompt: str,
        max_tokens_input: int
) -> Generator[tuple[str, str], None, None]:
    """Main debate function that orchestrates the conversation."""
    validate_inputs(topic, venice_key, max_tokens_input)

    try:
        yield "Initializing Client...", ""
        venice_client = get_venice_client(venice_key)
    except gr.Error as e:
        yield f"Initialization Error: {e}", "Error"
        return

    protagonist = DebateParticipant(
        model_name=protagonist_model_name,
        system_prompt=protagonist_system_prompt,
        is_protagonist=True
    )
    antagonist = DebateParticipant(
        model_name=antagonist_model_name,
        system_prompt=antagonist_system_prompt,
        is_protagonist=False
    )

    conversation_history: List[Dict[str, str]] = []
    debate_transcript = create_debate_header(topic, protagonist, antagonist, max_tokens_input)

    try:
        for turn in range(DEFAULT_DEBATE_TURNS * 2):
            current_participant = protagonist if turn % 2 == 0 else antagonist
            status_message = f"Turn {turn // 2 + 1} / {DEFAULT_DEBATE_TURNS} - {current_participant.display_name} {THINKING_ANIMATION[turn % 3]}"
            print(f"--- {status_message} ---")
            yield debate_transcript, status_message

            # Prepare message
            if turn == 0:
                current_user_instruction_text = f"Begin the debate by presenting your opening statement on the topic: '{topic}'."
            else:
                if not conversation_history:
                    raise gr.Error("Unexpected error: Conversation history is empty")
                last_message_content = conversation_history[-1]['content']
                current_user_instruction_text = (
                    f"Considering the debate history so far, present your response to the opponent's previous statement. "
                    f"Opponent's statement: '{textwrap.shorten(last_message_content, width=150, placeholder='...')}'"
                )

            messages = [
                {"role": "system", "content": current_participant.system_prompt},
                *conversation_history,
                {"role": "user", "content": current_user_instruction_text}
            ]

            try:
                print(f"Sending {len(messages)} messages to {current_participant.display_name}")

                stream = venice_client.chat.completions.create(
                    model=current_participant.model_name,
                    messages=messages,
                    max_tokens=max_tokens_input,
                    temperature=TEMPERATURE,
                    stream=True
                )

                debate_transcript += f"**{current_participant.display_name}:**\n"

                full_response, debate_transcript = yield from process_streaming_response(
                    stream, debate_transcript, status_message
                )

                # Update conversation history
                conversation_history.append({"role": "user", "content": current_user_instruction_text})
                ai_response_cleaned_text = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
                conversation_history.append({"role": "assistant", "content": ai_response_cleaned_text})

                debate_transcript += "\n\n---\n\n"
                time.sleep(1.4)

            except Exception as e:
                error_detail = str(e)
                if "AuthenticationError" in error_detail:
                    error_message = f"API Auth Error ({current_participant.display_name}). Check Key."
                elif "RateLimitError" in error_detail:
                    error_message = f"Rate Limit Error ({current_participant.display_name}). Wait & retry."
                elif "NotFoundError" in error_detail and "model" in error_detail:
                    error_message = f"Model Not Found ({current_participant.model_name}). Check Name/Access."
                elif ("BadRequestError" in error_detail and "context_length" in error_detail) or \
                        ("invalid_request_error" in error_detail and "maximum context length" in error_detail.lower()):
                    error_message = f"Context Length Exceeded ({current_participant.display_name}). Reduce turns/max_tokens or use a model with larger context."
                else:
                    error_message = f"API Error ({current_participant.display_name}): Check console."

                error_log_message = f"\n\n**API Error during {current_participant.display_name}'s turn:** {e}\nDebate halted."
                print(error_log_message)
                debate_transcript += f"**SYSTEM:**\n*{error_message}*\n\n---\n\n"
                yield debate_transcript, f"Error: {current_participant.display_name}"
                return

        status = "Debate Complete!"
        debate_transcript += f"**{status}**"
        print(status)
        yield debate_transcript, status

    except Exception as e:
        error_message = f"\n\n**An unexpected error occurred:** {e}\nDebate halted."
        print(error_message)
        debate_transcript += f"**SYSTEM:**\n*An unexpected error occurred: {e}*\n\n---\n\n"
        yield debate_transcript, "An unexpected error occurred."


# Clear any previous Gradio launches
gr.close_all()

# Define the Gradio Interface
with gr.Blocks(title="Debate - Venice AI",
               theme=gr.themes.Base(primary_hue="orange",
                                    font=[gr.themes.GoogleFont("Aeonik Fono"), "Arial", "sans-serif"])
                       .set(block_label_text_color="black",
                            block_label_text_color_dark="black",
                            ),
               css="""
    .settings-container {
        background: linear-gradient(135deg, #BEA989 0%, #EEEDE4 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
    }
    .gallery-item {
        background-color: var(--button-primary-background-fill);
    }
    .gallery.selected {
        background-color: var(--button-primary-background-fill-hover);
    }
    .chat-container {
        padding-top: 10px;
    }
    #example-debate-topics.label {
        color: black!important;
    }
    """) as demo:
    gr.Markdown(
        """
        # [<img src="https://venice.ai/images/icon-192.png" width="64"/>](https://venice.ai/images/icon-192.png) Venice Debate

        Set debate topic, API keys, models, max tokens, and custom system prompts (personas) for each AI.
        The prompts now encourage deeper engagement with the debate history for a more immersive experience.
        **Note:** Context limits can still be reached in long debates. Venice usage incurs costs.
        """
    )

    with gr.Row():
        # Left column - Settings
        with gr.Column(scale=1, elem_classes="settings-container"):
            topic_input = gr.Textbox(
                label="Debate Topic", placeholder="e.g., Should AI be uncensorable?", lines=2
            )

            # Examples
            gr.Examples(
                examples=[
                    ["Is anonymity on the internet a right or a privilege?"],
                    [
                        "Will the evolution of AI lead to more decentralized, privacy-focused models, or will centralized control become inevitable?"],
                    [
                        "Should AI models be allowed to use personal data without explicit consent to drive technological advancement?"]
                ],
                inputs=[topic_input], label="Example Debate Topics",
                elem_id='example-debate-topics'
            )

            config = Settings()
            gr.Markdown("### Credentials", visible=not bool(config.venice_key.get_secret_value()))

            venice_key_input = gr.Textbox(
                label="Venice API Key",
                type="password",
                placeholder="venice_api_key...",
                value=config.venice_key.get_secret_value(),
                visible=not bool(config.venice_key.get_secret_value())
            )

            gr.Markdown("### Model Selection")
            protagonist_model_input = gr.Dropdown(
                MODELS,
                value=DEFAULT_PROTAGONIST_MODEL,
                label="Protagonist Model Name"
            )
            antagonist_model_input = gr.Dropdown(
                MODELS,
                value=DEFAULT_ANTAGONIST_MODEL,
                label="Antagonist Model Name"
            )

            gr.Markdown("### Persona / System Prompts")
            protagonist_system_prompt_input = gr.Textbox(
                label="Protagonist System Prompt",
                placeholder="Define persona/role",
                value=DEFAULT_PROTAGONIST_SYS_PROMPT,
                lines=4
            )
            antagonist_system_prompt_input = gr.Textbox(
                label="Antagonist System Prompt",
                placeholder="Define persona/role",
                value=DEFAULT_ANTAGONIST_SYS_PROMPT,
                lines=4
            )

            gr.Markdown("### Debate Settings")
            max_tokens_slider = gr.Slider(
                label="Max Tokens per Turn",
                minimum=50,
                maximum=1700,
                step=10,
                value=DEFAULT_MAX_TOKENS
            )

            status_output = gr.Textbox(
                label="Status",
                placeholder="Waiting to start...",
                interactive=False
            )
            start_button = gr.Button("🚀 Start Debate", variant="primary")

        # Right column - Chat
        with gr.Column(scale=2, elem_classes="chat-container"):
            debate_output = gr.Markdown(
                label="Debate Transcript",
                value="*Debate transcript will appear here...*"
            )

    # --- Button Click Actions ---
    start_button.click(
        fn=run_debate,
        inputs=[
            topic_input,
            venice_key_input,
            protagonist_model_input,
            antagonist_model_input,
            protagonist_system_prompt_input,
            antagonist_system_prompt_input,
            max_tokens_slider
        ],
        outputs=[debate_output, status_output]
    )

# --- Launch App ---
demo.launch(share=True, debug=False, show_error=True, pwa=True)

print("\n✅ Gradio app launched!")
print("👉 Click the 'Running on public URL' link above.")
print("\n⚠️ Context window limits are still a possibility in long debates. Monitor token usage if needed.")
