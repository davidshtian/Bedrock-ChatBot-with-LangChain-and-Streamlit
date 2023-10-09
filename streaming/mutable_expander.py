from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.type_util import SupportsStr


class ChildType(Enum):
    """The enumerator of the child type."""

    MARKDOWN = "MARKDOWN"
    EXCEPTION = "EXCEPTION"


class ChildRecord(NamedTuple):
    """The child record as a NamedTuple."""

    type: ChildType
    kwargs: Dict[str, Any]
    dg: DeltaGenerator


class MutableExpander:
    """A Streamlit expander that can be renamed."""

    def __init__(self, parent_container: DeltaGenerator):
        """Create a new MutableExpander.

        Parameters
        ----------
        parent_container
            The `st.container` that the expander will be created inside.

            The expander transparently deletes and recreates its underlying
            `st.expander` instance when changes, and it uses
            `parent_container` to ensure it recreates this underlying expander in the
            same location onscreen.
        """
        self._parent_cursor = parent_container.empty()
        self._container = parent_container
        self._child_records: List[ChildRecord] = []

    def update(
        self,
    ) -> None:

        prev_records = self._child_records
        self._child_records = []

        # Replay all children into the new container
        for record in prev_records:
            self._create_child(record.type, record.kwargs)

    def markdown(
        self,
        body: SupportsStr,
        unsafe_allow_html: bool = False,
        *,
        help: Optional[str] = None,
        index: Optional[int] = None,
    ) -> int:
        """Add a Markdown element to the container and return its index."""
        kwargs = {"body": body,
                  "unsafe_allow_html": unsafe_allow_html, "help": help}
        new_dg = self._get_dg(index).markdown(
            **kwargs)  # type: ignore[arg-type]
        record = ChildRecord(ChildType.MARKDOWN, kwargs, new_dg)
        return self._add_record(record, index)

    def exception(
        self, exception: BaseException, *, index: Optional[int] = None
    ) -> int:
        """Add an Exception element to the container and return its index."""
        kwargs = {"exception": exception}
        new_dg = self._get_dg(index).exception(**kwargs)
        record = ChildRecord(ChildType.EXCEPTION, kwargs, new_dg)
        return self._add_record(record, index)

    def _create_child(self, type: ChildType, kwargs: Dict[str, Any]) -> None:
        """Create a new child with the given params"""
        if type == ChildType.MARKDOWN:
            self.markdown(**kwargs)
        elif type == ChildType.EXCEPTION:
            self.exception(**kwargs)
        else:
            raise RuntimeError(f"Unexpected child type {type}")

    def _add_record(self, record: ChildRecord, index: Optional[int]) -> int:
        """Add a ChildRecord to self._children. If `index` is specified, replace
        the existing record at that index. Otherwise, append the record to the
        end of the list.

        Return the index of the added record.
        """
        if index is not None:
            # Replace existing child
            self._child_records[index] = record
            return index

        # Append new child
        self._child_records.append(record)
        return len(self._child_records) - 1

    def _get_dg(self, index: Optional[int]) -> DeltaGenerator:
        if index is not None:
            # Existing index: reuse child's DeltaGenerator
            assert 0 <= index < len(self._child_records), f"Bad index: {index}"
            return self._child_records[index].dg

        # No index: use container's DeltaGenerator
        return self._container
