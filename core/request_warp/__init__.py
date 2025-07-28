import logging

logger = logging.getLogger(__name__)

from ._openai import OpenAIRequest
from .local import LocalRequest
from .wrapper import RequestWrapper

