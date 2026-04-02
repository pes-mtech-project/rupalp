import os
import httpx
import json
import subprocess
from abc import ABC
from typing import Callable, Union, Dict, Any, Union

### when use tgi model
api_key = '-' 

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


def build_completion_prompt(messages):
    prompt_lines = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            prompt_lines.append(f"System:\n{content}")
        elif role == "user":
            prompt_lines.append(f"User:\n{content}")
        else:
            prompt_lines.append(f"Assistant:\n{content}")
    prompt_lines.append("Assistant:\n")
    return "\n\n".join(prompt_lines)


class LongerThanContextError(Exception):
    pass

class ChatOpenAICompatible(ABC):
    def __init__(
        self,
        end_point: str,
        model="gemini-pro",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
    ):
        api_key = os.environ.get("OPENAI_API_KEY", "-")
        self.end_point = end_point
        self.model = model
        self.system_message = system_message
        self.other_parameters = {} if other_parameters is None else other_parameters
        self.is_legacy_completion = self.end_point.rstrip("/").endswith(
            "/v1/completions"
        )
        self.is_ollama_native_chat = self.end_point.rstrip("/").endswith("/api/chat")
        self.gemini_access_token = ""
        self.gemini_auth_mode = ""
        self.gemini_api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )

        
        if self.model.startswith("gemini-pro"):
            if "generativelanguage.googleapis.com" in self.end_point:
                if not self.gemini_api_key:
                    raise RuntimeError(
                        "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for "
                        "generativelanguage.googleapis.com endpoint."
                    )
                self.gemini_auth_mode = "api_key"
                self.headers = {"Content-Type": "application/json"}
            else:
                self.gemini_auth_mode = "vertex_bearer"
                self.gemini_access_token = self._fetch_gemini_access_token()
                self.headers = {
                    "Authorization": f"Bearer {self.gemini_access_token}",
                    "Content-Type": "application/json",
                }
        elif self.model.startswith("tgi"):
            self.headers = {
                        'Content-Type': 'application/json'
                    }   
        elif self.is_ollama_native_chat:
            self.headers = {"Content-Type": "application/json"}
        else:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

    def _fetch_gemini_access_token(self) -> str:
        access_token = os.environ.get("GEMINI_ACCESS_TOKEN", "").strip()
        if access_token:
            return access_token
        try:
            proc_result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "gcloud is not installed in this runtime. Install gcloud or set "
                "GEMINI_ACCESS_TOKEN."
            ) from e
        access_token = proc_result.stdout.strip()
        if access_token:
            return access_token
        raise RuntimeError(
            "Gemini auth token is empty. Run `gcloud auth login` and "
            "`gcloud auth application-default login` in this runtime, "
            "or set GEMINI_ACCESS_TOKEN."
        )

    def parse_response(self, response: httpx.Response) -> str:
        response_out = response.json()
        if self.model.startswith("gemini-pro"):
            return response_out["candidates"][0]["content"]["parts"][0]["text"]
        if self.model.startswith("tgi"):
            return response_out["generated_text"]
        if self.is_legacy_completion:
            return response_out["choices"][0]["text"].strip()
        if self.is_ollama_native_chat:
            return response_out["message"]["content"]
        # OpenAI-compatible APIs (OpenAI, Ollama /v1/chat/completions, etc.)
        if "choices" in response_out:
            return response_out["choices"][0]["message"]["content"]
        # Ollama /api/generate fallback.
        if "response" in response_out:
            return response_out["response"]
        raise NotImplementedError(f"Model {self.model} not implemented")

    def guardrail_endpoint(self) -> Callable:
        def end_point(input: str, **kwargs) -> str:
            input_str = [
                    # {"role": "system", "content": f"{self.system_message}"},
                    {"role": "system", "content": "You are a helpful assistant only capable of communicating with valid JSON, and no other text."},
                    {"role": "user", "content": f"{input}"},
                ]
            
            if self.model.startswith("gemini-pro"):
                if self.gemini_auth_mode == "vertex_bearer":
                    # Refresh token to avoid expiry on longer runs.
                    self.gemini_access_token = self._fetch_gemini_access_token()
                    self.headers["Authorization"] = f"Bearer {self.gemini_access_token}"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": input_str[1]["content"]}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "topP": 0.1,
                        "topK": 16,
                        "maxOutputTokens": 2048,
                        "candidateCount": 1,
                        "stopSequences": [],
                    },
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_LOW_AND_ABOVE",
                        }
                    ],
                }
                url = self.end_point
                if self.gemini_auth_mode == "api_key":
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}key={self.gemini_api_key}"
                response = httpx.post(url=url, headers=self.headers, json=payload, timeout=600.0)
                
            elif self.model.startswith("tgi"):
                llama_input_str = build_llama2_prompt(input_str)
                # print(llama_input_str)
                
                payload = {
                "inputs": llama_input_str,
                "parameters": {
                                "do_sample": True,
                                "top_p": 0.6,
                                "temperature": 0.8,
                                "top_k": 50,
                                "max_new_tokens": 256,
                                "repetition_penalty": 1.03,
                                "stop": ["</s>"]
                            }
                            }

                # payload = json.dumps(payload)
                response = httpx.post(
                    self.end_point, headers=self.headers, json=payload, timeout=600.0  # type: ignore
                )
            elif self.is_ollama_native_chat:
                payload = {
                    "model": self.model,
                    "messages": input_str,
                    "stream": False,
                }
                payload.update(self.other_parameters)
                response = httpx.post(
                    self.end_point, headers=self.headers, json=payload, timeout=600.0  # type: ignore
                )
            else:
                if self.is_legacy_completion:
                    payload = {
                        "model": self.model,
                        "prompt": build_completion_prompt(input_str),
                    }
                else:
                    payload = {
                        "model": self.model,
                        "messages": input_str,
                    }
                payload.update(self.other_parameters)

                response = httpx.post(
                    self.end_point, headers=self.headers, json=payload, timeout=600.0  # type: ignore
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if (response.status_code == 422) and ("must have less than" in response.text):
                    raise LongerThanContextError
                else:
                    raise e

            return self.parse_response(response)

        return end_point
