"""Callback Handler that prints to streamlit."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from mutable_expander import MutableExpander
from langchain.schema import LLMResult

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


def _convert_newlines(text: str) -> str:
    """Convert newline characters to markdown newline sequences
    (space, space, newline).
    """
    return text.replace("\n", "  \n")


class LLMThought:
    """A thought in the LLM's thought stream."""

    def __init__(
        self,
        parent_container: DeltaGenerator,
    ):
        """Initialize the LLMThought.

        Args:
            parent_container: The container we're writing into.
        """
        self._container = MutableExpander(
            parent_container=parent_container,
        )
        self._llm_token_stream = ""
        self._llm_token_writer_idx: Optional[int] = None

    @property
    def container(self) -> MutableExpander:
        """The container we're writing into."""
        return self._container

    def _reset_llm_token_stream(self) -> None:
        self._llm_token_stream = ""
        self._llm_token_writer_idx = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
        # self._reset_llm_token_stream()
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # This is only called when the LLM is initialized with `streaming=True`
        self._llm_token_stream += _convert_newlines(token)
        self._llm_token_writer_idx = self._container.markdown(
            self._llm_token_stream, index=self._llm_token_writer_idx
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # `response` is the concatenation of all the tokens received by the LLM.
        # If we're receiving streaming tokens from `on_llm_new_token`, this response
        # data is redundant
        self._reset_llm_token_stream()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._container.markdown("**LLM encountered an error...**")
        self._container.exception(error)

    def complete(self) -> None:
        """Finish the thought."""
        pass

    def clear(self) -> None:
        """Remove the thought from the screen. A cleared thought can't be reused."""
        self._container.clear()


class StreamlitCallbackHandler(BaseCallbackHandler):
    """A callback handler that writes to a Streamlit app."""

    def __init__(
        self,
        parent_container: DeltaGenerator,
    ):
        """Create a StreamlitCallbackHandler instance.

        Parameters
        ----------
        parent_container
            The `st.container` that will contain all the Streamlit elements that the
            Handler creates.
        """
        self._parent_container = parent_container
        self._current_thought: Optional[LLMThought] = None

    def _require_current_thought(self) -> LLMThought:
        """Return our current LLMThought. Raise an error if we have no current
        thought.
        """
        if self._current_thought is None:
            raise RuntimeError("Current LLMThought is unexpectedly None!")
        return self._current_thought

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThought(
                parent_container=self._parent_container,)

        self._current_thought.on_llm_start(serialized, prompts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_new_token(token, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_end(response, **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_error(error, **kwargs)

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        pass
