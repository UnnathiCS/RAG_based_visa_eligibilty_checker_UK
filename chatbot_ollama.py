"""
Simple Chatbot using Ollama (Open-source LLM)
This prototype uses Ollama to run a free LLM locally.

Installation:
1. Download Ollama from https://ollama.ai
2. Run: ollama pull mistral (or another model like llama2, neural-chat)
3. Install required packages: pip install requests

To run this bot, make sure Ollama is running in the background.
"""

import requests
import json

class SimpleOllamaChatbot:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        """
        Initialize chatbot with Ollama.
        
        Args:
            model_name: The model to use (e.g., 'mistral', 'llama2', 'neural-chat')
            base_url: Ollama server URL (default is localhost:11434)
        """
        self.model = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def chat(self, user_message):
        """
        Send message to the LLM and get response.
        
        Args:
            user_message: The user's input text
            
        Returns:
            The LLM's response text
        """
        payload = {
            "model": self.model,
            "prompt": user_message,
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Sorry, I couldn't process that.")
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running on http://localhost:11434"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def start_conversation(self):
        """Start interactive conversation loop."""
        print("=== Ollama Chatbot ===")
        print("Type 'exit' to quit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nChatbot: Processing...", end="", flush=True)
            response = self.chat(user_input)
            print(f"\rChatbot: {response}\n")


if __name__ == "__main__":
    # Initialize and run the chatbot
    chatbot = SimpleOllamaChatbot(model_name="mistral")
    chatbot.start_conversation()
