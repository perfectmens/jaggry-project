import subprocess
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def ask_ollama(prompt, model_name="llama3.2:3b"):
    """Ask Ollama model via CLI and get response (UTF-8 safe)."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,                # pass as str, NOT bytes
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,                   # text=True ensures str input/output
            encoding="utf-8",            # decode output as UTF-8
            errors="replace"             # replace invalid characters
        )
        if result.returncode != 0:
            return f"Ollama CLI Error: {result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"



# Example usage
response = ask_ollama("Hello! How are you?", model_name="llama3.2:3b")
print(response)
