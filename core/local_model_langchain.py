#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng (Modified for ChatOpenAI compatibility)
# [Descriptions] : LangChain implementation for local LLM models
# ==================================================================
import requests
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Mapping, Tuple
from func_timeout import func_set_timeout
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from dataclasses import dataclass, field
from cachetools import TTLCache, cachedmethod
import operator
from cachetools.keys import hashkey

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import Field, root_validator, Extra
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    ChatGenerationChunk,
    LLMResult,
)
from langchain_core.runnables import RunnableConfig

from log import logger
from openai import OpenAI  # 引入 OpenAI 客户端


@dataclass
class ModelConfig:
    """Configuration for LLM models"""

    url: str
    max_len: int
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 20
    min_p: int = 0
    retry_attempts: int = 10
    timeout: int = 200
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)
    openai_client: Optional[Any] = None

# Model configurations
MODEL_CONFIGS = {
    "Qwen25-72B": ModelConfig(
        url="http://0.0.0.0:9071/v1/chat/completions",
        max_len=5800,
    ),
    "Qwen25-7B": ModelConfig(
        url="http://0.0.0.0:9072/v1/chat/completions",
        max_len=8192,
    ),
    "llama3-70b": ModelConfig(
        url="http://0.0.0.0:9087/v1/chat/completions",
        max_len=131072,
    ),
    "Qwen3-32B": ModelConfig(
        url="http://0.0.0.0:9094/v1",
        max_len=131072,
        openai_client=OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:9094/v1",
        ),
    ),
}



class RequestsClient:
    """HTTP client with retry logic and caching"""

    def __init__(self):
        self.session = self._create_session()
        self.cache = TTLCache(maxsize=100, ttl=3600)  # Cache responses for 1 hour

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic"""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        return session

    def _cache_key(self, url: str, data: Dict[str, Any], timeout: int) -> tuple:
        """Custom cache key function that makes data hashable"""
        # Convert data dict to a sorted tuple of items to ensure consistency
        data_str = json.dumps(data, sort_keys=True)  # Serialize dict to string
        return hashkey(url, data_str, timeout)  # Use cachetools' hashkey

    @cachedmethod(operator.attrgetter("cache"), key=_cache_key)
    def make_request(
        self, url: str, data: Dict[str, Any], timeout: int
    ) -> Optional[str]:
        """Make HTTP request with caching"""
        try:
            if "Qwen3-32B" in data["model"]:  # 判断是否使用 Qwen3-32B 模型
                return self._make_qwen3_request(data)
            else:
                response = self.session.post(url, json=data, timeout=timeout)
                response.raise_for_status()
                response = response.json()
                if response and "choices" in response:
                    response_text = response["choices"][0]["message"]["content"]
                    # Process response for r1 models with think tags
                    if "</think>" in response_text:
                        response_text = response_text.split("</think>")[-1]
                return response_text
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def _make_qwen3_request(self, data: Dict[str, Any]) -> Optional[str]:
        """Handle requests specifically for Qwen3-32B using OpenAI client"""
        logger.info("_make_qwen3_request")
        try:
            openai_client = MODEL_CONFIGS[data["model"]].openai_client
            chat_response = openai_client.chat.completions.create(
                model="Qwen/Qwen3-32B",
                messages=data["messages"],
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.8),
                presence_penalty=1.5,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": data.get("top_k", 20),  # Added to extra_body
                    "min_p": data.get("min_p", 0),  # Added to extra_body
                },  # Disable thinking mode
            )
            msg = chat_response.choices[0].message.reasoning_content
            if msg:
                if "</think>" in msg:
                    msg = msg.split("</think>")[-1]
            return msg
        except Exception as e:
            logger.error(
                f"Qwen3-32B request failed: {str(e)}--{traceback.format_exc()}"
            )
            return None

    def stream_request(
        self, url: str, data: Dict[str, Any], timeout: int
    ) -> Iterator[Dict[str, Any]]:
        """Make streaming HTTP request"""
        try:
            # Set stream to True in the data
            data["stream"] = True

            response = self.session.post(url, json=data, timeout=timeout, stream=True)
            response.raise_for_status()

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Skip the "data: " prefix if present
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse line: {line}")
        except Exception as e:
            logger.error(f"Streaming request failed: {str(e)}")
            logger.error(traceback.format_exc())


Callbacks = Optional[Union[List[Callable], Callable]]


class LocalChatModel(BaseChatModel):
    """LangChain chat model implementation for local LLM models compatible with ChatOpenAI"""

    # client: Any = Field(default_factory=RequestsClient)
    client: RequestsClient = None

    # 仅作为类型提示
    model_name: str
    temperature: float
    model_kwargs: Dict[str, Any]

    # Streaming and retries
    streaming: bool = False
    max_retries: int = 6

    # Response parameters
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None

    # Custom LocalChatModel parameters
    base_url: Optional[str] = None  # Can override the default URL from MODEL_CONFIGS
    request_timeout: Optional[float] = None

    def __init__(
        self,
        model_name: str = "Qwen25-7B",
        temperature: float = 0.7,
        model_kwargs: Optional[Dict[str, Any]] = None,
        streaming: bool = False,
        max_retries: int = 6,
        # 其他参数...
        **kwargs,
    ):
        # Initialize base properties but don't create client here
        model_kwargs = model_kwargs or {}

        # Call parent class initializer
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
            max_retries=max_retries,
            model_kwargs=model_kwargs,  # Pass it here
            **kwargs,
        )
        self.client = RequestsClient()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the model exists in our configurations."""
        model_name = values["model_name"]

        if not isinstance(values.get("model_kwargs", {}), dict):
            values["model_kwargs"] = {}

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model name {model_name} not found in MODEL_CONFIGS. "
                f"Available models: {', '.join(MODEL_CONFIGS.keys())}"
            )

        # Initialize timeout from config if not provided
        if values.get("request_timeout") is None:
            values["request_timeout"] = MODEL_CONFIGS[model_name].timeout

        # Set max_tokens from model config if not provided
        if values.get("max_tokens") is None:
            values["max_tokens"] = MODEL_CONFIGS[model_name].max_len

        return values

    @property
    def _llm_type(self) -> str:
        return "local_chat_model"

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format LangChain messages to the format expected by the API"""
        formatted_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(
                    {"role": "system", "content": message.content}
                )
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            elif isinstance(message, ChatMessage):
                formatted_messages.append(
                    {"role": message.role, "content": message.content}
                )
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

        # Handle special cases for r1 and qwq models
        if "r1" in self.model_name or "qwq" in self.model_name:
            if (
                len(formatted_messages) >= 2
                and formatted_messages[0]["role"] == "system"
            ):
                system = formatted_messages[0]["content"]
                prompt = formatted_messages[1]["content"]
                formatted_messages = [
                    {
                        "role": "user",
                        "content": f"{system}\n\n{prompt}\n<think>\n",
                    }
                ]

        # Ensure think tag for r1 models
        if (
            "r1" in self.model_name
            and formatted_messages
            and ("<think>" not in formatted_messages[-1]["content"])
        ):
            formatted_messages[-1]["content"] += "\n<think>\n"

        return formatted_messages

    def _prepare_request_data(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Prepare the request data for the API"""
        config = MODEL_CONFIGS[self.model_name]

        # Start with base parameters
        data = {
            "model": self.model_name,
            "messages": messages,
            "tools": [],
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "n": kwargs.get("n", self.n),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens or config.max_len),
            "stream": kwargs.get("stream", self.streaming),
        }

        # Add stop sequences if provided
        stop = kwargs.get("stop") or self.stop
        if stop:
            data["stop"] = stop

        # Add frequency and presence penalties if model supports them
        frequency_penalty = kwargs.get("frequency_penalty", self.frequency_penalty)
        if frequency_penalty > 0:
            data["frequency_penalty"] = frequency_penalty

        presence_penalty = kwargs.get("presence_penalty", self.presence_penalty)
        if presence_penalty > 0:
            data["presence_penalty"] = presence_penalty

        # Add any model-specific kwargs
        if hasattr(self.model_kwargs, "items"):  # 检查是否有items方法
            for k, v in self.model_kwargs.items():
                if k not in data:
                    data[k] = v

        # Add any additional kwargs from config
        for k, v in config.additional_kwargs.items():
            if k not in data:
                data[k] = v

        return data

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate chat completion"""
        formatted_messages = self._format_messages(messages)

        # Override stop sequence if provided
        if stop:
            kwargs["stop"] = stop

        # Get model config and prepare request data
        config = MODEL_CONFIGS[self.model_name]
        data = self._prepare_request_data(formatted_messages, **kwargs)

        # Determine the URL to use
        url = self.base_url or config.url
        timeout = self.request_timeout or config.timeout

        logger.info(f"Requesting {self.model_name} at {url}")

        # Handle streaming mode
        if data.get("stream", False):
            return self._generate_stream(
                formatted_messages=formatted_messages,
                run_manager=run_manager,
                url=url,
                timeout=timeout,
                data=data,
            )

        # Make request with retries
        response_text = None
        for attempt in range(self.max_retries):
            try:
                response_text = self.client.make_request(url, data, timeout)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All attempts failed for {self.model_name}")
                    logger.error(traceback.format_exc())

        if response_text is None:
            logger.error("Failed to get response from model")
            response_text = ""

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _generate_stream(
        self,
        formatted_messages: List[Dict[str, str]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        url: str = None,
        timeout: int = None,
        data: Dict[str, Any] = None,
    ) -> ChatResult:
        """Generate streaming chat completion"""
        chunks = []
        chunk_text = ""

        # Process the stream
        try:
            for chunk in self.client.stream_request(url, data, timeout):
                if not chunk or "choices" not in chunk:
                    continue

                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content = delta["content"]

                    # Process response for r1 models with think tags
                    if "</think>" in content:
                        content = content.split("</think>")[-1]

                    chunk_text += content

                    # Create a chunk for callback
                    message_chunk = AIMessageChunk(content=content)
                    generation_chunk = ChatGenerationChunk(message=message_chunk)
                    chunks.append(generation_chunk)

                    if run_manager:
                        run_manager.on_llm_new_token(content)
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            logger.error(traceback.format_exc())

        # Create the final message
        if "</think>" in chunk_text:
            chunk_text = chunk_text.split("</think>")[-1]

        message = AIMessage(content=chunk_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Asynchronously generate chat completion"""
        # For now, use the synchronous implementation
        # This could be improved with async HTTP client in the future
        return self._generate(
            messages, stop, run_manager and run_manager.get_sync(), **kwargs
        )

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate multiple completions"""
        # This implementation matches ChatOpenAI's behavior
        return super().generate(messages, stop, callbacks, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text string."""
        # Simple approximation - this should be improved with actual tokenization
        # based on the specific model
        return len(text) // 4  # Very rough approximation

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in a list of messages."""
        formatted_messages = self._format_messages(messages)
        total_tokens = 0
        for message in formatted_messages:
            total_tokens += self.get_num_tokens(message["content"])
            # Add a small overhead for message formatting
            total_tokens += 4
        return total_tokens


# Compatibility class for AIMessageChunk
class AIMessageChunk(AIMessage):
    """AIMessage chunk for streaming."""


# Compatibility type for callbacks
Callbacks = Optional[Union[List[Callable], Callable]]


# Utility function to get a response from a model with timeout
@func_set_timeout(200)
def get_from_llm(
    messages: Union[str, List[Dict[str, str]], List[BaseMessage]],
    model_name: str = "Qwen25-7B",
    **kwargs,
) -> Optional[str]:
    """
    Get response from LLM using LangChain

    Args:
        messages: Input messages (string, list of dicts, or LangChain messages)
        model_name: Name of the model to use
        **kwargs: Additional parameters to override defaults

    Returns:
        Generated response or None if failed
    """
    try:
        # Initialize the model
        model = LocalChatModel(model_name=model_name, **kwargs)

        # Convert messages to LangChain format if needed
        if isinstance(messages, str):
            langchain_messages = [HumanMessage(content=messages)]
        elif isinstance(messages, list) and all(
            isinstance(m, BaseMessage) for m in messages
        ):
            langchain_messages = messages
        else:
            # Convert dict messages to LangChain messages
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(ChatMessage(role=role, content=content))

        # Get response
        result = model.invoke(langchain_messages)
        return result.content
    except Exception as e:
        logger.error(f"Error in get_from_llm: {str(e)}")
        logger.error(traceback.format_exc())
        return None


LOCAL_MODELS = {
    model_name: LocalChatModel(
        model_name=model_name, stop=["\n\n"], presence_penalty=0.1
    )
    for model_name in MODEL_CONFIGS
}

# if __name__ == "__main__":
#     # Example 1: Simple message
#     model = LocalChatModel(model_name="Qwen3-32B", temperature=0.5, max_tokens=2000)
#     response = model.invoke("Tell me about artificial intelligence")
#     print(f"s1 Response: {response.content}")

#     # Example 3: Using ChatOpenAI-style parameters
#     messages = [
#         SystemMessage(content="You are a helpful assistant"),
#         HumanMessage(content="Write a short poem about technology"),
#     ]
#     for model_name in MODEL_CONFIGS:
#         model = LocalChatModel(model_name=model_name, temperature=0.5, max_tokens=2000)
#         response = model.invoke(messages)
#         print(f"Response from {model_name}: {response.content}")
