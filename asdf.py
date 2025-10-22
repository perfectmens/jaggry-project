import subprocess
import json

def ask_ollama(prompt):
    """
    Ask Ollama via CLI and get response.
    """
    try:
        # Run ollama chat with model "your_model_name"
        result = subprocess.run(
            ["ollama", "chat", "your_model_name", "--prompt", prompt, "--json"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return f"Ollama CLI Error: {result.stderr}"
        
        # Parse JSON output
        response = json.loads(result.stdout)
        return response.get("completion", "")
    except Exception as e:
        return f"Error: {e}"

# Example usage
answer = ask_ollama("Hello, how are you?")
print(answer)
