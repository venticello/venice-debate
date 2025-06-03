# Venice Debate

An AI-powered debate platform that enables two AI models to engage in structured debates on various topics. The platform uses Venice AI models to create engaging and thought-provoking discussions.

## Features

- Real-time streaming debate responses
- Customizable AI personas (Protagonist and Antagonist)
- Multiple model options from Venice AI
- Adjustable response length
- Beautiful and responsive UI
- Support for think-aloud reasoning (using `<think>` tags)

## Prerequisites

- Python 3.8+
- Venice AI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/venticello/venice-debate.git
cd venice-debate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Venice AI API key:
```
VENICE_KEY=your_api_key_here
```
If the key is not specified in the environment (.env), it can be entered in the settings.
## Usage

1. Start the application:
```bash
python debate.py
```

2. Open your browser and navigate to the provided URL (usually http://localhost:7860)

3. Configure the debate:
   - Enter a debate topic
   - Select models for Protagonist and Antagonist
   - Customize system prompts (optional)
   - Adjust max tokens per turn
   - Click "Start Debate"

## Configuration

### Environment Variables
- `VENICE_KEY`: Your Venice AI API key

### Available Models
- venice-uncensored
- qwen3-235b
- deepseek-r1-671b
- llama-3.1-405b
- llama-3.3-70b
- dolphin-2.9.2-qwen2-72b
- qwen-2.5-qwq-32b
- mistral-31-24b

## Features in Detail

### Think-Aloud Reasoning
The AI models can use `<think>` tags to show their reasoning process. These thoughts are displayed in blockquotes in the debate transcript.

### Custom Personas
You can customize the behavior of both AI participants by modifying their system prompts. The default prompts encourage:
- Protagonist: Analytical and precise debating style
- Antagonist: Creative and insightful challenging approach

### Real-time Streaming
Responses are streamed in real-time, providing an engaging experience as you watch the debate unfold.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Venice AI for providing the AI models
- Gradio for the web interface framework 