#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "claude-agent-sdk>=0.1.0",
#   "httpx",
#   "ipython>=8.0.0",
#   "matplotlib",
#   "pygments",
#   "trio",
#   "tyro>=0.9.0",
# ]
# ///
"""
cc_jupyter — Claude Code magic for IPython and Jupyter notebooks
================================================================

Provides ``%cc`` and ``%%cc`` line/cell magics that launch a full Claude Code
agent to write code into your notebook cells.  You review and run each proposed
cell — human-in-the-loop by design.

Claude Code is invoked via ``claude-agent-sdk`` (spawns the ``claude`` CLI).
No Anthropic API key is needed — authentication is handled by your existing
``claude auth login`` session.


INSTALLATION
------------
From the repo in editable/dev mode::

    pip install -e /path/to/claude_jupyter

Or install the wheel directly::

    pip install cc_jupyter-0.2.1-py3-none-any.whl

With uv (no prior install needed — runs the self-test)::

    uv run cc_jupyter.py


AUTOLOADER — run on every IPython start
----------------------------------------
Copy the startup script so the magic loads automatically::

    cp startup/00-claude-autoload.py \\
       "$(ipython locate profile default)/startup/00-claude-autoload.py"

Or load it manually in any session::

    %load_ext cc_jupyter


USAGE
-----
::

    %cc plot a sine wave with matplotlib
    %%cc
    Load 'data.csv', drop nulls, and print df.describe()
    %cc_new                           # fresh conversation
    %cc --model opus <prompt>         # stronger model
    %cc --import myfile.py <prompt>   # add file as context
    %cc --add-dir ./data              # grant directory access
    %cc --mcp-config .mcp.json        # attach MCP servers
    %cc --max-cells 1 <prompt>        # limit cells per turn
    %cc --clean                       # hide prompt cells
    %cc --help                        # all flags


PYDOC
-----
::

    python -m pydoc cc_jupyter
    python -m pydoc cc_jupyter.ClaudeJupyterMagics


REQUIREMENTS
------------
- Python >= 3.11
- IPython >= 8.0
- ``claude`` CLI installed and authenticated

Install the Claude CLI if you don't have it::

    # macOS / Linux
    curl -fsSL https://claude.ai/install.sh | bash

    # Windows (PowerShell)
    irm https://claude.ai/install.ps1 | iex

Then log in::

    claude auth login
"""

# =============================================================================
# REGENERATIVE PROMPT
# =============================================================================
# You are regenerating cc_jupyter.py — a single-file IPython/Jupyter magic that
# integrates Claude Code into notebooks via claude-agent-sdk.
#
# The file must contain, in order:
#   1. PEP 723 uv script header with dependencies
#   2. Module docstring (install, usage, pydoc instructions)
#   3. This regenerative prompt as a comment block
#   4. All source merged in dependency order:
#        constants → capture_helpers → CellWatcher → VariableTracker →
#        HistoryManager → ConfigManager → prompt_builder functions →
#        CellQueueManager + jupyter_integration functions →
#        ClaudeClientManager → execute_python_tool (@tool) →
#        ClaudeJupyterMagics → load_ipython_extension
#   5. __main__ block with standalone smoke tests (no pytest)
#
# Key constraints:
#   - claude-agent-sdk spawns the claude CLI; no ANTHROPIC_API_KEY needed
#   - Human-in-the-loop: Claude proposes cells, user executes them
#   - Thread-safe cell queue via threading.RLock
#   - History capped at MAX_HISTORY_CELLS = 100
#   - IPython 8+ compatible: use run_line_magic(), never .magic()
#   - _magic_instance must remain a module-level variable (required by @tool)
# =============================================================================

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from time import monotonic
from typing import Any

import trio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
from IPython import get_ipython as _get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

__version__ = "0.2.1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXECUTE_PYTHON_TOOL_NAME = "mcp__jupyter__create_python_cell"

try:
    import importlib.util
    PYGMENTS_AVAILABLE = importlib.util.find_spec("pygments") is not None
except Exception:
    PYGMENTS_AVAILABLE = False

HELP_TEXT = """
🚀 Claude Code Magic loaded!
Features:
  • Full agentic Claude Code execution
  • Cell-based code approval workflow
  • Real-time message streaming
  • Session state preservation
  • Conversation continuity across cells

Usage:
  %cc <instructions>       # Continue with additional instructions (one-line)
  %%cc <instructions>      # Continue with additional instructions (multi-line)
  %cc_new (or %ccn)        # Start fresh conversation
  %cc --help               # Show available options and usage information

Context management:
  %cc --import <file>       # Add a file to be included in initial conversation messages
  %cc --add-dir <dir>       # Add a directory to Claude's accessible directories
  %cc --mcp-config <file>   # Set path to a .mcp.json file containing MCP server configurations
  %cc --cells-to-load <num> # The number of cells to load into a new conversation (default: all for first %cc, none for %cc_new)

Output:
  %cc --model <name>       # Model to use for Claude Code (default: sonnet)
  %cc --max-cells <num>    # Set the maximum number of cells CC can create per turn (default: 3)

Display:
  %cc --clean              # Replace prompt cells with Claude's code cells
  %cc --no-clean           # Turn off the above setting (default)

When to use each form:
  • Use %cc (single %) for short, one-line instructions
  • Use %%cc (double %) for multi-line instructions or detailed prompts

Notes:
- Restart the kernel to stop the Claude session
"""

QUEUED_EXECUTION_TEXT = """
⚠️ Not executing this prompt because you've queued multiple cell executions (e.g. Run All),
so re-running Claude might be unintentional. If you did mean to do this, please add the
flag `--allow-run-all` and try again.
"""

CLEANUP_PROMPTS_TEXT = """
🧹 Persistent preference set. For the rest of this session, cells with prompts will {maybe_not}
be cleaned up after executing.
"""

MAX_HISTORY_CELLS = 100

# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def extract_images_from_captured(captured_output: Any) -> list[dict[str, Any]]:
    """Extract image data from an IPython ``capture_output()`` context object."""
    images: list[dict[str, Any]] = []
    if not hasattr(captured_output, "outputs") or not captured_output.outputs:
        return images
    for output in captured_output.outputs:
        if hasattr(output, "data") and isinstance(output.data, dict):
            for img_format in ["image/png", "image/jpeg", "image/jpg", "image/svg+xml"]:
                if img_format in output.data:
                    image_info: dict[str, Any] = {
                        "format": img_format,
                        "data": output.data[img_format],
                        "metadata": getattr(output, "metadata", {}),
                    }
                    if img_format in image_info["metadata"]:
                        image_info["dimensions"] = image_info["metadata"][img_format]
                    images.append(image_info)
    return images


def format_images_summary(images: list[dict[str, Any]]) -> str:
    """Return a human-readable summary of captured images for inclusion in prompts."""
    if not images:
        return ""
    lines = ["The following images were captured from the code execution:"]
    for i, img in enumerate(images, 1):
        dims = ""
        if "dimensions" in img:
            d = img["dimensions"]
            if isinstance(d, dict) and "width" in d and "height" in d:
                dims = f" ({d['width']}x{d['height']})"
        preview = img["data"][:50] + "..." if len(img["data"]) > 50 else img["data"]
        lines += [f"\nImage {i}:", f"  Format: {img['format']}{dims}", f"  Base64 preview: {preview}"]
    lines.append("\nNote: Full image data is available in the captured output.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cell watcher
# ---------------------------------------------------------------------------

QUEUED_EXECUTION_THRESHOLD_SECONDS = 0.1


class CellWatcher:
    """Watches cell execution timing to detect Run-All queued execution."""

    def __init__(self) -> None:
        self.last_cell_finish_time = monotonic()
        self.time_between_cell_executions: deque[float] = deque(maxlen=2)

    def pre_run_cell(self, info: Any) -> None:
        self.time_between_cell_executions.append(monotonic() - self.last_cell_finish_time)

    def post_run_cell(self, result: Any) -> None:
        if result.execution_count:
            self.last_cell_finish_time = monotonic()

    def was_execution_probably_queued(self) -> bool:
        if len(self.time_between_cell_executions) < 2:
            return False
        previous_gap, current_gap = self.time_between_cell_executions
        return (
            previous_gap < QUEUED_EXECUTION_THRESHOLD_SECONDS
            and current_gap < QUEUED_EXECUTION_THRESHOLD_SECONDS
        )


# ---------------------------------------------------------------------------
# Variable tracker
# ---------------------------------------------------------------------------

class VariableTracker:
    """Tracks and reports changes in IPython session variables."""

    def __init__(self, shell: InteractiveShell | None) -> None:
        self.shell = shell
        self._previous_variables: dict[str, Any] = {}

    def reset(self) -> None:
        self._previous_variables = {}

    def get_truncated_repr(self, value: Any, max_length: int = 100) -> str:
        try:
            r = repr(value)
            return r[:max_length - 3] + "..." if len(r) > max_length else r
        except Exception:
            return f"<{type(value).__name__} object>"

    def get_variables_info(self) -> str:
        """Return a formatted diff of session variables since last call."""
        try:
            if self.shell is None:
                return "The IPython session has no user-defined variables."
            user_ns = self.shell.user_ns
            filtered = {
                k: v for k, v in user_ns.items()
                if not k.startswith("_") and k not in ["In", "Out", "exit", "quit"]
            }
            if not filtered and not self._previous_variables:
                return "The IPython session has no user-defined variables."

            added = [n for n in filtered if n not in self._previous_variables]
            modified = [
                n for n, v in filtered.items()
                if n in self._previous_variables
                and self.get_truncated_repr(v) != self._previous_variables[n]
            ]
            removed = [n for n in self._previous_variables if n not in filtered]

            self._previous_variables = {n: self.get_truncated_repr(v) for n, v in filtered.items()}

            lines: list[str] = []
            if added:
                lines.append("New variables:")
                for n in sorted(added):
                    lines.append(f"  + {n}: {type(filtered[n]).__name__} = {self.get_truncated_repr(filtered[n])}")
            if modified:
                if lines:
                    lines.append("")
                lines.append("Modified variables:")
                for n in sorted(modified):
                    lines.append(f"  ~ {n}: {type(filtered[n]).__name__} = {self.get_truncated_repr(filtered[n])}")
            if removed:
                if lines:
                    lines.append("")
                lines.append("Removed variables:")
                for n in sorted(removed):
                    lines.append(f"  - {n}")

            if not lines:
                return "No variable changes detected since last interaction."
            return "Variable changes in IPython session:\n" + "\n".join(lines)
        except Exception:
            return "Could not retrieve session variables."


# ---------------------------------------------------------------------------
# History manager
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages IPython history tracking and formatting for Claude context."""

    def __init__(self, shell: InteractiveShell | None) -> None:
        self.shell = shell
        self.last_output_line = 0

    def reset_output_tracking(self) -> None:
        self.last_output_line = 0

    def update_last_output_line(self) -> None:
        if self.shell is not None:
            self.last_output_line = len(self.shell.user_ns.get("In", [])) - 1

    def get_history_range(self, start: int | None = None, stop: int | None = None) -> list[tuple[int, int, Any]]:
        if self.shell is None or self.shell.history_manager is None:
            return []
        try:
            hm = self.shell.history_manager
            if not hm.input_hist_parsed or len(hm.input_hist_parsed) <= 1:
                return []
            return list(hm.get_range(
                session=0, start=start, stop=stop, raw=False, output=True,
            ))
        except Exception as e:
            _logger.debug(f"Failed to get history range: {e}")
            return []

    def format_cell(self, line_num: int, input_code: str, output_result: Any = None) -> str:
        """Format a cell as XML tags for Claude context."""
        parts = [f"<cell-in-{line_num}>", input_code.strip(), f"</cell-in-{line_num}>"]
        if output_result is not None:
            parts += [
                f"<cell-out-{line_num}>",
                output_result if isinstance(output_result, str) else repr(output_result),
                f"</cell-out-{line_num}>",
            ]
        return "\n".join(parts)

    def get_shell_output_since_last(self) -> str:
        """Return cells executed since last Claude call, formatted for context."""
        try:
            interactions: list[str] = []
            history = self.get_history_range(start=self.last_output_line + 1, stop=None)
            if history:
                for _, line_num, item in history:
                    inp, out = item if isinstance(item, tuple) else (item, None)
                    if inp and not inp.strip().startswith("get_ipython().run_cell_magic"):
                        cell = self.format_cell(line_num, inp, out)
                        if out is None and self.shell:
                            out2 = self.shell.user_ns.get("Out", {}).get(line_num)
                            if out2 is not None:
                                cell = self.format_cell(line_num, inp, out2)
                        interactions.append(cell)
            elif self.shell:
                in_list = self.shell.user_ns.get("In", [])
                out_dict = self.shell.user_ns.get("Out", {})
                for i in range(self.last_output_line + 1, len(in_list)):
                    cmd = in_list[i] if i < len(in_list) else None
                    if cmd and not cmd.strip().startswith("get_ipython().run_cell_magic"):
                        interactions.append(self.format_cell(i, cmd, out_dict.get(i)))

            if interactions:
                return (
                    "\nRecent IPython cell executions "
                    "(Note: Only return values are captured, print statements are not shown):\n"
                    + "\n".join(interactions) + "\n"
                )
            return ""
        except Exception as e:
            _logger.warning(f"Failed to get shell output: {e}")
            return ""

    def get_last_executed_cells(self, n: int) -> str:
        """Return the last *n* cells as formatted context (capped at MAX_HISTORY_CELLS)."""
        if n == 0:
            return ""
        if n == -1:
            history = self.get_history_range(start=-MAX_HISTORY_CELLS, stop=None)
        elif n > 0:
            if n > MAX_HISTORY_CELLS:
                _logger.warning(f"Requested {n} cells exceeds recommended limit of {MAX_HISTORY_CELLS}")
            history = self.get_history_range(start=-n, stop=None)
        else:
            return ""
        try:
            if not history:
                return ""
            cells = ["Last executed cells from this session:"]
            for _sid, line_num, item in history:
                inp, out = item if isinstance(item, tuple) else (item, None)
                if inp and not inp.strip().startswith("get_ipython().run_cell_magic"):
                    cells.append(self.format_cell(line_num, inp, out))
            return "\n\n".join(cells) if len(cells) > 1 else ""
        except Exception as e:
            _logger.warning(f"Failed to format cell history: {e}")
            return ""


# ---------------------------------------------------------------------------
# Config manager
# ---------------------------------------------------------------------------

class ConfigManager:
    """Manages all configuration state for the Claude Code magic."""

    def __init__(self) -> None:
        self.should_cleanup_prompts = False
        self.editing_current_cell = False
        self.is_new_conversation: bool = True
        self.is_current_execution_verbose: bool = False
        self.max_cells = 3
        self.create_python_cell_count = 0
        self.model = "sonnet"
        self.imported_files: list[str] = []
        self.added_directories: list[str] = []
        self.mcp_config_file: str | None = None
        self.cells_to_load: int = -1
        self.cells_to_load_user_set: bool = False
        self.timeout_seconds: int = 300

    def reset_for_new_conversation(self) -> None:
        self.is_new_conversation = True
        self.create_python_cell_count = 0
        if not self.cells_to_load_user_set:
            self.cells_to_load = 0

    def handle_cc_options(self, args: Any, cell_watcher: CellWatcher) -> bool:
        """Process flags from %cc. Returns True if the magic should return early."""
        if args.help:
            print(HELP_TEXT)
            return True

        if args.clean is not None:
            self.should_cleanup_prompts = args.clean
            print(CLEANUP_PROMPTS_TEXT.format(maybe_not="" if self.should_cleanup_prompts else "not "))
            return True

        pickup = (
            "Will be used in next new conversation."
            if self.is_new_conversation
            else "Use %cc_new to pick up the setting."
        )

        if args.max_cells is not None:
            old = self.max_cells
            self.max_cells = args.max_cells
            print(f"📝 Set max_cells from {old} to {self.max_cells}. {pickup}")
            return True

        if args.import_file is not None:
            fp = Path(args.import_file).expanduser().resolve()
            try:
                if not fp.exists():
                    print(f"❌ Import failed: File not found: {fp}")
                    return True
                if not fp.is_file():
                    print(f"❌ Import failed: Not a file: {fp}")
                    return True
                with fp.open("rb") as f:
                    if b"\x00" in f.read(1024):
                        print(f"❌ Import failed: {fp.name} appears to be binary")
                        return True
                s = str(fp)
                if s not in self.imported_files:
                    self.imported_files.append(s)
                    print(f"✅ Added {fp.name} to import list. {pickup}")
                else:
                    print(f"ℹ️ {fp} is already in the import list.")
            except PermissionError:
                print(f"❌ Import failed: Permission denied: {fp}")
            except Exception as e:
                print(f"❌ Import failed: {fp.name}: {e}")
            return True

        if args.add_dir is not None:
            dp = Path(args.add_dir).expanduser().resolve()
            try:
                if not dp.exists():
                    print(f"❌ Directory not found: {dp}")
                    return True
                if not dp.is_dir():
                    print(f"❌ Not a directory: {dp}")
                    return True
                s = str(dp)
                if s not in self.added_directories:
                    self.added_directories.append(s)
                    print(f"✅ Added {dp} to accessible directories. {pickup}")
                else:
                    print(f"ℹ️ {dp} is already in the accessible directories list.")
            except PermissionError:
                print(f"❌ Permission denied: {dp}")
            except Exception as e:
                print(f"❌ Error accessing directory {dp}: {e}")
            return True

        if args.mcp_config is not None:
            cp = Path(args.mcp_config).expanduser().resolve()
            try:
                if not cp.exists():
                    print(f"❌ MCP config not found: {cp}")
                    return True
                with cp.open() as f:
                    try:
                        json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON in MCP config: {e}")
                        return True
                self.mcp_config_file = str(cp)
                print(f"✅ Set MCP config to {cp}. {pickup}")
            except Exception as e:
                print(f"❌ Error loading MCP config {cp}: {e}")
            return True

        if args.model is not None:
            self.model = args.model
            print(f"✅ Set model to {self.model}. {pickup}")
            return True

        if args.cells_to_load is not None:
            if args.cells_to_load < -1:
                print("❌ Number of cells must be -1 (all), 0 (none), or positive")
                return True
            self.cells_to_load = args.cells_to_load
            self.cells_to_load_user_set = True
            if args.cells_to_load == 0:
                print("✅ Disabled loading recent cells for new conversations")
            elif args.cells_to_load == -1:
                print("✅ Will load all available cells for new conversations (capped at 100)")
            else:
                print(f"✅ Will load up to {args.cells_to_load} recent cell(s) for new conversations")
            return True

        if cell_watcher.was_execution_probably_queued() and not args.allow_run_all:
            print(QUEUED_EXECUTION_TEXT)
            return True

        return False

    def get_claude_code_options_settings(self) -> str | None:
        if self.added_directories:
            return json.dumps({"permissions": {"additionalDirectories": self.added_directories}})
        return None

    def get_mcp_servers(self, mcp_server_script: str) -> dict[str, Any]:
        mcp_servers: dict[str, Any] = {}
        if mcp_server_script:
            mcp_servers["local_executor"] = {"command": "python", "args": [mcp_server_script]}
        if self.mcp_config_file:
            try:
                with Path(self.mcp_config_file).open() as f:
                    data = json.load(f)
                    if "mcpServers" in data and isinstance(data["mcpServers"], dict):
                        mcp_servers.update(data["mcpServers"])
            except Exception as e:
                print(f"⚠️ Error loading MCP config {self.mcp_config_file}: {e}")
        return mcp_servers


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def get_system_prompt(is_ipython: bool, max_cells: int) -> str:
    """Generate the system prompt for Claude based on the runtime environment."""
    if is_ipython:
        env = "shared IPython session"
        tool_call_result = f"The {EXECUTE_PYTHON_TOOL_NAME} tool call will populate the next input with the Python code you provide."
        preference = (
            f"You can only call {EXECUTE_PYTHON_TOOL_NAME} once, since the IPython terminal does not allow "
            "for multiple pending code blocks.\n\nThe user will see the code block and can choose to execute it or not."
        )
    else:
        env = "Jupyter notebook"
        tool_call_result = f"Each {EXECUTE_PYTHON_TOOL_NAME} call will create a new cell in the user's Jupyter notebook interface."
        preference = (
            f"IMPORTANT: Prefer to call {EXECUTE_PYTHON_TOOL_NAME} only ONCE with a short code snippet.\n"
            f"As a last resort, you may call it multiple times to split up a large code block. "
            f"You can make at most {max_cells} calls per turn (i.e., in response to each user prompt).\n"
            "The user will be presented with the code blocks one by one.\n"
            "If the user executes it and it succeeds, the next code block gets created in a new cell.\n"
            "If the user executes it and it errors, then the error will get reported, but the chain is broken. "
            "Assume that the user does not see the subsequent code.\n"
            "If the user executes a cell that is not the next code block, then the chain will pause until the proper next code block is executed.\n\n"
            f"If the user asks you to modify code in the current cell, you may do this by using the {EXECUTE_PYTHON_TOOL_NAME} tool EXACTLY ONCE.\n"
            "Identifying that the current cell is the target is obvious because the code block is included directly in the user's request itself.\n\n"
            f"If the user asks you to edit/change/modify code in a DIFFERENT cell, inform them that you do not have that capability.\n"
            f"Instead, suggest that they use `%%cc edit this cell to <requested edits>` at the top of the cell they would like to edit.\n"
            f"Respond ONLY with that suggestion. DO NOT create new cells for the request and DEFINITELY DO NOT use the {EXECUTE_PYTHON_TOOL_NAME} tool."
        )

    preamble = (
        f"You are operating in a {env}.\n"
        f"You can see the current session state. You can create new code cells using the {EXECUTE_PYTHON_TOOL_NAME} tool.\n"
        f"{tool_call_result}\n"
        f"Never call {EXECUTE_PYTHON_TOOL_NAME} if you can answer the user's question directly with text.\n"
        f"{preference}\n"
    )
    image_capture = (
        "IMPORTANT: When generating code that displays images (matplotlib, seaborn, PIL, etc.), "
        "you MUST wrap that code with IPython's capture_output() and the variable `_claude_captured_output` "
        "to capture the images. Then, you must re-display the captured output. You can only have one "
        "_claude_captured_output context. Use this pattern:\n\n"
        "```\n"
        "from IPython.utils.capture import capture_output\n\n"
        "import matplotlib.pyplot as plt\n\n"
        "with capture_output() as _claude_captured_output:\n"
        "    plt.plot([1, 2, 3, 4])\n"
        "    plt.show()\n\n"
        "for output in _claude_captured_output.outputs:\n"
        "    display(output)\n"
        "```\n\n"
        "This allows the system to capture any images for you for further processing."
    )
    tool_usage = (
        f"For any questions you can answer on your own, DO NOT use {EXECUTE_PYTHON_TOOL_NAME}.\n"
        f"Don't forget that you have other built-in tools like Read. Try responding using your built-in tools first "
        f"without using {EXECUTE_PYTHON_TOOL_NAME}. Your response does not need to invoke {EXECUTE_PYTHON_TOOL_NAME}.\n"
        f"If you want to explain something to the user, do not put your explanation in {EXECUTE_PYTHON_TOOL_NAME}. "
        "Just return regular prose.\n\n"
        "IMPORTANT: Do not invoke {EXECUTE_PYTHON_TOOL_NAME} in parallel.\n"
        "IMPORTANT: Always include a return value or expression at the end of your "
        f"{EXECUTE_PYTHON_TOOL_NAME} output. Only return values are captured in output cells - "
        "print statements are NOT captured.\n"
        "For example, instead of print(df.head()), use df.head() as the last line.\n\n"
        "If <request> is empty, it is because the user wants you to continue from where you left off."
    ).format(EXECUTE_PYTHON_TOOL_NAME=EXECUTE_PYTHON_TOOL_NAME)
    return "\n".join([preamble, image_capture, tool_usage])


def prepare_imported_files_content(imported_files: list[str]) -> str:
    """Return file contents formatted for inclusion in a Claude conversation."""
    if not imported_files:
        return ""
    parts = []
    for path_str in imported_files:
        fp = Path(path_str)
        if fp.exists():
            try:
                parts.append(f"{fp.name}:\n```\n{fp.read_text()}\n```")
            except Exception:
                pass
    if parts:
        return (
            "Files imported by the user for your reference. Use this content directly. Don't read them again:\n\n"
            + "\n\n".join(parts)
        )
    return ""


def build_enhanced_prompt(
    prompt: str,
    variables_info: str,
    previous_execution: str = "",
    shell_output: str = "",
    is_new_conversation: bool = False,
    imported_files_content: str = "",
    last_cells_content: str = "",
    captured_images: list[dict[str, Any]] | None = None,
) -> str | list[dict[str, Any]]:
    """Build the prompt sent to Claude, optionally including images."""
    if captured_images is None:
        captured_images = []

    text = (
        f"\nYour client's request is <request>{prompt}</request>\n\n"
        f"{variables_info}\n{previous_execution}\n"
    )
    if shell_output:
        text += shell_output

    if is_new_conversation:
        ctx = [p for p in [imported_files_content, last_cells_content] if p]
        if ctx:
            text = "\n\n".join(ctx) + "\n\n" + text

    if captured_images:
        blocks: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": img["format"], "data": img["data"]}}
            for img in captured_images
        ]
        blocks.append({"type": "text", "text": text})
        return blocks

    return text


# ---------------------------------------------------------------------------
# Jupyter integration
# ---------------------------------------------------------------------------

def is_in_jupyter_notebook() -> bool:
    """Return True when running inside a Jupyter notebook kernel."""
    ip = _get_ipython()
    return ip is not None and hasattr(ip, "kernel")


class CellQueueManager:
    """Thread-safe queue of Claude-generated cells awaiting user execution."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue: list[dict[str, Any]] = []

    def add_cell(self, cell_info: dict[str, Any]) -> int:
        with self._lock:
            self._queue.append(cell_info)
            return len(self._queue)

    def get_queue(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._queue.copy()

    def get_next_unexecuted(self) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            for i, cell in enumerate(self._queue):
                if not cell.get("executed", False):
                    return i, cell
            return None

    def mark_executed(self, index: int, had_exception: bool = False) -> None:
        with self._lock:
            if 0 <= index < len(self._queue):
                self._queue[index]["executed"] = True
                self._queue[index]["had_exception"] = had_exception

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def update_cell(self, index: int, updates: dict[str, Any]) -> None:
        with self._lock:
            if 0 <= index < len(self._queue):
                self._queue[index].update(updates)


def create_approval_cell(
    parent: Any,
    code: str,
    request_id: str,
    should_cleanup_prompts: bool,
    tool_use_id: str | None = None,
) -> None:
    """Create a cell for user approval and add it to the queue."""
    marker_id = tool_use_id if tool_use_id else request_id
    marker = f"# Claude cell [{marker_id}]"
    marked_code = f"{marker}\n{code}"

    queue_position = 0
    if parent.shell is not None:
        parent.shell.user_ns["_claude_request_id"] = request_id
        if not hasattr(parent, "_cell_queue_manager"):
            parent._cell_queue_manager = CellQueueManager()
        queue_position = parent._cell_queue_manager.add_cell({
            "code": marked_code,
            "original_code": code,
            "tool_use_id": tool_use_id,
            "request_id": request_id,
            "marker_id": marker_id,
            "marker": marker,
            "executed": False,
        })

    print("\n" + "=" * 60, flush=True)
    print("🤖 Claude wants to execute code", flush=True)
    print("-" * 60, flush=True)

    if parent._config_manager.is_current_execution_verbose:
        if PYGMENTS_AVAILABLE:
            from pygments import highlight
            from pygments.formatters import TerminalFormatter
            from pygments.lexers import PythonLexer
            print(highlight(marked_code, PythonLexer(), TerminalFormatter()), flush=True)
        else:
            print(marked_code, flush=True)
        print("-" * 60, flush=True)

    if queue_position == 1:
        if parent.shell is not None:
            parent.shell.user_ns["_claude_pending_input"] = marked_code
        if is_in_jupyter_notebook():
            loc = "above" if should_cleanup_prompts else "below"
            print(f"📋 To approve: Run the cell {loc}", flush=True)
        else:
            print("📋 Press Enter to execute the code (or edit it first)", flush=True)
    else:
        print("⏳ Queued: Cell will appear automatically after previous cell executes", flush=True)

    print("➡️ To continue Claude agentically afterward: Run %cc", flush=True)
    print("=" * 60 + "\n", flush=True)


def adjust_cell_queue_markers(parent: Any) -> None:
    """Update cell markers with decorated borders after all cells are queued."""
    if parent.shell is None or not hasattr(parent, "_cell_queue_manager"):
        return
    cell_queue = parent._cell_queue_manager.get_queue()
    if not cell_queue:
        return

    n = len(cell_queue)
    for i, cell_info in enumerate(cell_queue):
        original_code = cell_info["original_code"]
        mid = cell_info["marker_id"]

        title = f"# ║                        CLAUDE GENERATED CELL [id: {mid}]"
        if len(title) < 80:
            title += " " * (79 - len(title)) + "║"

        pos = ""
        if n > 1:
            pos_line = f"# ║                                 CELL {i + 1} OF {n}"
            pos = ("\n" + pos_line + " " * (79 - len(pos_line)) + "║") if len(pos_line) < 80 else ("\n" + pos_line)

        marker = (
            "# ╔════════════════════════════════════════════════════════════════════════════╗\n"
            f"{title}{pos}\n"
            "# ╚════════════════════════════════════════════════════════════════════════════╝\n"
        )
        parent._cell_queue_manager.update_cell(i, {"code": f"{marker}\n{original_code}", "marker": marker})
        if i == 0 and parent.shell is not None:
            parent.shell.user_ns["_claude_pending_input"] = f"{marker}\n{original_code}"


def process_cell_queue(parent: Any) -> None:
    """Advance the cell queue after a successful execution."""
    if parent.shell is None or not hasattr(parent, "_cell_queue_manager"):
        return
    next_cell = parent._cell_queue_manager.get_next_unexecuted()
    if next_cell is not None:
        idx, cell_info = next_cell
        parent.shell.set_next_input(cell_info.get("code", ""))
        remaining = sum(
            1 for c in parent._cell_queue_manager.get_queue()[idx:]
            if not c.get("executed", False)
        )
        if remaining > 0:
            print(f"📋 Next cell ready (Claude cell [{cell_info['marker_id']}])", flush=True)
    else:
        q = parent._cell_queue_manager.get_queue()
        if len(q) > 1 and all(c.get("executed", False) for c in q):
            if any(c.get("had_exception", False) for c in q):
                print("⚠️ All of Claude's generated cells processed (some with errors)", flush=True)
            else:
                print("✅ All of Claude's generated cells have been processed successfully", flush=True)


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

MARKDOWN_PATTERNS = ["```", "`", "    ", "\t", "**", "##", "](", "---", ">", "~~"]


def _display_claude_message(text: str) -> None:
    msg = f"💭 Claude: {text}"
    if not is_in_jupyter_notebook():
        print(msg, flush=True)
        return
    has_markdown = any(p in text for p in MARKDOWN_PATTERNS)
    if not has_markdown:
        print(msg)
        return
    try:
        from IPython.display import Markdown, display
        display(Markdown(msg))
    except ImportError:
        print(msg, flush=True)


def _format_tool_call(name: str, inp: dict[str, Any]) -> str:
    names = {"LS": "List", "GrepToolv2": "Search", EXECUTE_PYTHON_TOOL_NAME: "CreateNotebookCell"}
    dn = names.get(name, name)
    if name == "Read":
        parts = [f"{dn}({inp.get('file_path', '')})"]
        if "offset" in inp: parts.append(f"offset: {inp['offset']}")
        if "limit" in inp: parts.append(f"limit: {inp['limit']}")
        return " ".join(parts)
    if name == "LS":
        return f"{dn}({inp.get('path', '')})"
    if name in ["Write", "Edit", "MultiEdit"]:
        return f"{dn}({inp.get('file_path', '')})"
    if name == "Bash":
        return f'{dn}("{inp.get("command", "")}")'
    if name == "Glob":
        p, path = inp.get("pattern", ""), inp.get("path", "")
        return f'{dn}(pattern: "{p}", path: "{path}")' if path else f'{dn}("{p}")'
    if name == "GrepToolv2":
        parts = [f'{dn}(pattern: "{inp.get("pattern", "")}"', f'path: "{inp.get("path")}"']
        if "glob" in inp: parts.append(f'glob: "{inp["glob"]}"')
        return ", ".join(parts) + ")"
    if name == "WebFetch":
        return f'{dn}("{inp.get("url", "")}")'
    if name == "WebSearch":
        return f'{dn}("{inp.get("query", "")}")'
    return dn


class ClaudeClientManager:
    """Manages ClaudeSDKClient instances — creates a fresh client per query."""

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._interrupt_requested: bool = False
        self._current_client: ClaudeSDKClient | None = None

    async def query_sync(
        self,
        prompt: str | list[dict[str, Any]],
        options: ClaudeAgentOptions,
        is_new_conversation: bool,
        verbose: bool = False,
        enable_interrupt: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Send a query and collect all responses. Creates a fresh client each call."""
        await trio.lowlevel.checkpoint()
        tool_calls: list[str] = []
        assistant_messages: list[str] = []
        self._interrupt_requested = False

        if self._session_id and not is_new_conversation:
            if not options.resume:
                options.resume = self._session_id
            options.continue_conversation = True

        client = ClaudeSDKClient(options=options)
        self._current_client = client

        try:
            await client.connect()

            if isinstance(prompt, list):
                @trio.as_safe_channel
                async def _gen() -> Any:
                    await trio.lowlevel.checkpoint()
                    yield {"type": "user", "message": {"role": "user", "content": prompt}, "parent_tool_use_id": None}
                    await trio.lowlevel.checkpoint()
                async with _gen() as ch:
                    await client.query(ch)
            else:
                await client.query(prompt)

            has_printed_model = not is_new_conversation
            messages_to_process: list[Any] = []

            if enable_interrupt:
                async with trio.open_nursery() as nursery:
                    async def _collect() -> None:
                        await trio.lowlevel.checkpoint()
                        async for msg in client.receive_response():
                            messages_to_process.append(msg)
                            if isinstance(msg, ResultMessage):
                                break
                    nursery.start_soon(_collect)
                    while True:
                        if self._interrupt_requested:
                            nursery.cancel_scope.cancel()
                            await client.interrupt()
                            print("\n⚠️ Query interrupted by user", flush=True)
                            break
                        if messages_to_process and isinstance(messages_to_process[-1], ResultMessage):
                            break
                        await trio.sleep(0.05)
            else:
                async for msg in client.receive_response():
                    messages_to_process.append(msg)
                    if isinstance(msg, ResultMessage):
                        break

            for msg in messages_to_process:
                if isinstance(msg, AssistantMessage):
                    if hasattr(msg, "model") and not has_printed_model:
                        print(f"🧠 Claude model: {msg.model}")
                        has_printed_model = True
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            _display_claude_message(block.text)
                            assistant_messages.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            print(f"⏺ {_format_tool_call(block.name, block.input)}", flush=True)
                            if verbose:
                                print(f"  ⎿  Arguments: {block.input}", flush=True)
                            tool_calls.append(f"{block.name}: {block.input}")
                elif isinstance(msg, ResultMessage):
                    if msg.session_id and msg.session_id != self._session_id:
                        self._session_id = msg.session_id
                        print(f"📍 Claude Code Session ID: {self._session_id}", flush=True)

        except Exception as e:
            err = str(type(e)) + str(e)
            if any(x in err for x in ["BrokenResourceError", "BrokenPipeError", "ClosedResourceError"]):
                if not self._interrupt_requested:
                    print("\n⚠️ Connection lost. A new connection will be created automatically.", flush=True)
            else:
                print(f"\n❌ Error during Claude execution: {e!s}")
                if verbose:
                    print(traceback.format_exc())
        finally:
            try:
                with trio.CancelScope(shield=True, deadline=trio.current_time() + 2):
                    await client.disconnect()
            except Exception:
                pass
            self._current_client = None

        return assistant_messages, tool_calls

    async def handle_interrupt(self) -> None:
        self._interrupt_requested = True
        if self._current_client is not None:
            with contextlib.suppress(Exception):
                await self._current_client.interrupt()
        await trio.lowlevel.checkpoint()

    def reset_session(self) -> None:
        self._session_id = None

    @property
    def session_id(self) -> str | None:
        return self._session_id


async def run_streaming_query(
    parent: Any,
    prompt: str | list[dict[str, Any]],
    options: ClaudeAgentOptions,
    verbose: bool,
) -> None:
    if not hasattr(parent, "_client_manager") or parent._client_manager is None:
        parent._client_manager = ClaudeClientManager()
    await parent._client_manager.query_sync(
        prompt, options, parent._config_manager.is_new_conversation, verbose
    )
    parent._history_manager.update_last_output_line()


# ---------------------------------------------------------------------------
# Magic commands
# ---------------------------------------------------------------------------

# Module-level ref required by the @tool function (which runs in a trio thread).
_magic_instance: ClaudeJupyterMagics | None = None  # type: ignore[name-defined]


@tool(
    "create_python_cell",
    "Create a cell with Python code in the IPython environment",
    {"code": str},
)
async def execute_python_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Handle create_python_cell tool calls — create cells and return immediately."""
    if _magic_instance is None:
        await trio.lowlevel.checkpoint()
        return {"content": [{"type": "text", "text": "❌ Magic instance not initialized"}], "is_error": True}

    code = args.get("code", "")
    if not code:
        await trio.lowlevel.checkpoint()
        return {"content": [{"type": "text", "text": "❌ No code provided"}], "is_error": True}

    cfg = _magic_instance._config_manager
    if cfg.create_python_cell_count >= cfg.max_cells:
        await trio.lowlevel.checkpoint()
        return {
            "content": [{"type": "text", "text": (
                f"❌ Maximum number of cells ({cfg.max_cells}) reached for this turn. "
                "Please wait for the user to provide additional input."
            )}],
            "is_error": True,
        }

    tool_use_id = str(uuid.uuid4())
    try:
        request_id = _magic_instance.current_request_id
        if not request_id:
            request_id = str(uuid.uuid4())
            _magic_instance.current_request_id = request_id
        if request_id not in _magic_instance.pending_requests:
            _magic_instance.pending_requests[request_id] = {"timestamp": time.time()}

        _magic_instance._create_approval_cell(code, request_id, tool_use_id)
        cfg.create_python_cell_count += 1
        await trio.lowlevel.checkpoint()
        return {"content": [{"type": "text", "text": (
            "✅ Code cell created. Waiting for user to review and execute. "
            "The user will run %cc when ready to proceed."
        )}]}
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        await trio.lowlevel.checkpoint()
        return {"content": [{"type": "text", "text": f"❌ Error creating cell: {e!s}"}], "is_error": True}


@magics_class
class ClaudeJupyterMagics(Magics):
    """IPython magic providing ``%cc`` / ``%%cc`` Claude Code integration.

    Registers ``%cc``, ``%%cc``, ``%cc_new``, and ``%ccn`` in IPython.
    Each invocation spawns a Claude Code agent that proposes notebook cells;
    the user reviews and executes each cell manually.

    Instantiated automatically by ``load_ipython_extension``.
    """

    def __init__(self, shell: InteractiveShell, cell_watcher: CellWatcher) -> None:
        super().__init__(shell)
        global _magic_instance  # noqa: PLW0603
        _magic_instance = self

        self.cell_watcher = cell_watcher
        self._variable_tracker = VariableTracker(shell)
        self._history_manager = HistoryManager(shell)
        self._config_manager = ConfigManager()
        self._cell_queue_manager = CellQueueManager()
        self.pending_requests: dict[str, dict[str, Any]] = {}
        self.current_request_id: str | None = None
        self._client_manager: ClaudeClientManager | None = None

        if shell is not None:
            shell.events.register("post_run_cell", self._post_run_cell_hook)

        from IPython.core.inputtransformer2 import EscapedCommand, HelpEnd
        HelpEnd.priority = EscapedCommand.priority + 1

    def _create_approval_cell(self, code: str, request_id: str, tool_use_id: str | None = None) -> None:
        should_cleanup = self._config_manager.should_cleanup_prompts or self._config_manager.editing_current_cell
        create_approval_cell(self, code, request_id, should_cleanup, tool_use_id)

    def _post_run_cell_hook(self, result: Any) -> None:
        if self.shell is None:
            return
        cell_queue = self._cell_queue_manager.get_queue()
        if not cell_queue:
            return

        last_input = self.shell.user_ns.get("In", [""])[-1] if "In" in self.shell.user_ns else ""

        next_idx: int | None = None
        next_marker: str | None = None
        next_marker_id: str | None = None
        for i, cell_info in enumerate(cell_queue):
            if not cell_info["executed"]:
                next_idx = i
                next_marker = cell_info.get("marker", "")
                next_marker_id = cell_info["marker_id"]
                break

        executed_expected = bool(next_marker and last_input.startswith(next_marker))

        if next_idx is not None and executed_expected:
            had_exc = not result.success if result else False
            self._cell_queue_manager.mark_executed(next_idx, had_exception=had_exc)
            if result and not result.success and result.error_in_exec:
                self._cell_queue_manager.update_cell(next_idx, {
                    "error": {"type": type(result.error_in_exec).__name__, "message": str(result.error_in_exec)}
                })

        if executed_expected and result and result.success:
            process_cell_queue(self)
        elif executed_expected and result and not result.success:
            remaining = sum(1 for c in self._cell_queue_manager.get_queue() if not c.get("executed", False))
            if remaining > 0:
                print(f"\n⚠️ Execution failed. {remaining} cell(s) remaining in queue.", flush=True)
                print("Run %cc to continue with the error in context, or %cc_new to start fresh.", flush=True)
        elif not executed_expected and next_marker:
            for cell_info in cell_queue:
                m = cell_info.get("marker", "")
                if m and last_input.startswith(m):
                    print(f"\n⚠️ Claude cell [{cell_info['marker_id']}] executed out of order. "
                          f"Expected [{next_marker_id}] next.", flush=True)
                    print("Run the expected cell to continue, or %cc to report results.", flush=True)
                    break

    def __del__(self) -> None:
        global _magic_instance  # noqa: PLW0603
        _magic_instance = None
        if self.shell is not None:
            with contextlib.suppress(Exception):
                self.shell.events.unregister("post_run_cell", self._post_run_cell_hook)
        self._client_manager = None

    def _execute_prompt(
        self,
        prompt: str,
        previous_execution: str = "",
        captured_images: list[dict[str, Any]] | None = None,
        verbose: bool = False,
    ) -> None:
        if captured_images is None:
            captured_images = []

        self.current_request_id = str(uuid.uuid4())
        self._config_manager.create_python_cell_count = 0

        if self.shell is not None and "_claude_captured_output" in self.shell.user_ns:
            captured_output = self.shell.user_ns.pop("_claude_captured_output")
            captured_images = extract_images_from_captured(captured_output)

        variables_info = self._variable_tracker.get_variables_info()
        shell_output = "" if self._config_manager.is_new_conversation else self._history_manager.get_shell_output_since_last()

        text = (
            f"\nYour client's request is <request>{prompt}</request>\n\n"
            f"{variables_info}\n{previous_execution}\n"
        )
        if shell_output:
            text += shell_output

        if self._config_manager.is_new_conversation:
            ctx: list[str] = []
            if self._config_manager.imported_files:
                imported = prepare_imported_files_content(self._config_manager.imported_files)
                if imported:
                    ctx.append(imported)
            if self._config_manager.cells_to_load != 0:
                last_cells = self._history_manager.get_last_executed_cells(self._config_manager.cells_to_load)
                if last_cells:
                    ctx.append(last_cells)
            if ctx:
                text = "\n\n".join(ctx) + "\n\n" + text

        enhanced_prompt: str | list[dict[str, Any]]
        if captured_images:
            print(format_images_summary(captured_images), flush=True)
            blocks: list[dict[str, Any]] = [
                {"type": "image", "source": {"type": "base64", "media_type": img["format"], "data": img["data"]}}
                for img in captured_images
            ]
            blocks.append({"type": "text", "text": text})
            enhanced_prompt = blocks
        else:
            enhanced_prompt = text

        sdk_server = create_sdk_mcp_server(name="jupyter_executor", version="1.0.0", tools=[execute_python_tool])
        mcp_servers: dict[str, Any] = {"jupyter": sdk_server}
        additional = self._config_manager.get_mcp_servers("")
        if additional:
            mcp_servers.update(additional)

        options = ClaudeAgentOptions(
            allowed_tools=[
                "Bash", "LS", "Grep", "Read", "Edit", "MultiEdit", "Write",
                "WebSearch", "WebFetch", "mcp__jupyter__create_python_cell",
            ],
            model=self._config_manager.model,
            mcp_servers=mcp_servers,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": get_system_prompt(
                    is_ipython=not is_in_jupyter_notebook(),
                    max_cells=self._config_manager.max_cells,
                ),
            },
            setting_sources=["user", "project", "local"],
        )

        settings_json = self._config_manager.get_claude_code_options_settings()
        if settings_json:
            options.settings = settings_json

        try:
            if Path("/root/code").exists():
                options.cwd = "/root/code"
        except (PermissionError, OSError):
            pass

        if self._client_manager is not None and self._client_manager.session_id:
            options.resume = self._client_manager.session_id

        exc_queue: queue.Queue[Exception] = queue.Queue()

        def _run() -> None:
            try:
                trio.run(self._run_streaming_query, enhanced_prompt, options, verbose)
            except Exception as e:
                exc_queue.put(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        original_handler = None

        def _sigint(signum: int, frame: Any) -> None:
            if self._client_manager is not None:
                print("Interrupting Claude…", flush=True)
                def _interrupt() -> None:
                    if self._client_manager is not None:
                        with contextlib.suppress(Exception):
                            trio.run(self._client_manager.handle_interrupt)
                threading.Thread(target=_interrupt, daemon=True).start()

        try:
            original_handler = signal.signal(signal.SIGINT, _sigint)
            thread.join()
        finally:
            if original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)

        if not exc_queue.empty():
            raise exc_queue.get()

        self._history_manager.update_last_output_line()
        adjust_cell_queue_markers(self)

        if self.shell is not None and "_claude_pending_input" in self.shell.user_ns:
            pending = self.shell.user_ns.pop("_claude_pending_input")
            with contextlib.suppress(Exception):
                self.shell.set_next_input(
                    pending,
                    replace=(self._config_manager.should_cleanup_prompts or self._config_manager.editing_current_cell),
                )

        self._config_manager.is_new_conversation = False

    async def _run_streaming_query(
        self, prompt: str | list[dict[str, Any]], options: ClaudeAgentOptions, verbose: bool
    ) -> None:
        self._config_manager.is_current_execution_verbose = verbose
        await run_streaming_query(self, prompt, options, verbose)
        self._config_manager.is_current_execution_verbose = False

    def _claude_continue_impl(self, request_id: str, additional_prompt: str = "", verbose: bool = False) -> str:
        cell_queue = self._cell_queue_manager.get_queue()
        if verbose:
            executed = sum(1 for c in cell_queue if c.get("executed", False))
            print(f"📊 Cell summary: {executed}/{len(cell_queue)} executed", flush=True)

        results: list[str] = []
        for i, cell in enumerate(cell_queue):
            tool_use_id = cell.get("tool_use_id", "")
            code = cell.get("original_code", cell.get("code", ""))
            executed = cell.get("executed", False)
            had_exc = cell.get("had_exception", False)
            error_info = cell.get("error", None)
            output = None

            if executed:
                with contextlib.suppress(Exception):
                    for _sid, _ln, (inp, out) in self._history_manager.get_history_range(start=-10, stop=None):
                        if inp.strip() == code.strip() and out is not None:
                            output = str(out)
                            break

            prefix = f"Tool use {tool_use_id}: " if tool_use_id else f"Cell {i + 1}: "
            if executed:
                if had_exc:
                    err = f"{error_info['type']}: {error_info['message']}" if error_info else "an error"
                    prefix += f"Executed but encountered {err}"
                elif output:
                    prefix += f"Executed successfully with output:\n{output}"
                else:
                    prefix += "Executed successfully (no output)"
            else:
                prefix += "Not executed by user"
            results.append(prefix)

        continue_prompt = "Previous code execution results:\n" + "\n\n".join(results)
        additional_prompt = additional_prompt.strip() or "Please continue with the task."
        print("✅ Continuing Claude session with execution results…", flush=True)

        if self.shell is not None:
            self.shell.user_ns.pop("_claude_request_id", None)
            self.shell.user_ns.pop("_claude_cell_queue", None)
        self._cell_queue_manager.clear()
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]

        self._execute_prompt(additional_prompt, continue_prompt)
        return request_id

    def _handle_cc_options(self, args: Any) -> bool:
        return self._config_manager.handle_cc_options(args, self.cell_watcher)

    def _parse_args_and_prompt(self, line: str, magic_func: Any) -> tuple[Any, str]:
        self._config_manager.editing_current_cell = False
        parts = line.split(None, 1) if line else []
        if not parts:
            return parse_argstring(magic_func, ""), ""
        if parts[0].startswith("-"):
            first = parts[0]
            remaining = parts[1] if len(parts) > 1 else ""
            value_args: list[str] = []
            for action in magic_func.parser._actions:
                if action.option_strings and action.nargs != 0:
                    value_args.extend(action.option_strings)
            if first in value_args:
                vp = remaining.split(None, 1) if remaining else []
                if vp:
                    return parse_argstring(magic_func, f"{first} {vp[0]}"), (vp[1] if len(vp) > 1 else "")
                return parse_argstring(magic_func, first), ""
            return parse_argstring(magic_func, first), remaining
        if any("=" in ln or "(" in ln for ln in line.splitlines()[1:]):
            self._config_manager.editing_current_cell = True
        return parse_argstring(magic_func, ""), line

    @line_cell_magic
    @magic_arguments()
    @argument("--verbose", "-v", action="store_true")
    @argument("--allow-run-all", "-a", action="store_true")
    @argument("--clean", action=argparse.BooleanOptionalAction, default=None)
    @argument("--max-cells", type=int, default=None)
    @argument("--help", "-h", action="store_true")
    @argument("--import", type=str, dest="import_file")
    @argument("--add-dir", type=str, dest="add_dir")
    @argument("--mcp-config", type=str, dest="mcp_config")
    @argument("--cells-to-load", type=int, dest="cells_to_load")
    @argument("--model", type=str, dest="model")
    def cc(self, line: str, cell: str | None = None) -> None:
        """Run Claude Code with full agentic loop.

        Usage (line magic)::

            %cc create a fibonacci function
            %cc --verbose
            %cc --help

        Usage (cell magic)::

            %%cc
            Create a pandas DataFrame and plot it

        Note: if your prompt ends with '?', use ``%%cc`` to avoid IPython's help system.
        """
        if cell is not None:
            line = line + "\n" + cell
        args, prompt = self._parse_args_and_prompt(line, self.cc)
        if self._handle_cc_options(args):
            return
        if not prompt:
            return
        request_id = self.shell.user_ns.get("_claude_request_id") if self.shell else None
        if request_id:
            self._claude_continue_impl(request_id, prompt, args.verbose)
            return
        if self.shell is not None:
            stale = self._cell_queue_manager.get_queue()
            unexecuted = sum(1 for c in stale if not c.get("executed", False))
            if unexecuted > 0:
                print(f"⚠️ Clearing {unexecuted} unexecuted cells from previous request", flush=True)
            self._cell_queue_manager.clear()
            self.shell.user_ns.pop("_claude_cell_queue", None)
        if not prompt:
            raise ValueError("A prompt must be provided to start the conversation.")
        self._execute_prompt(prompt, verbose=args.verbose)

    @line_cell_magic
    def ccn(self, line: str, cell: str | None = None) -> None:
        """Alias for ``%cc_new``."""
        self.cc_new(line, cell)

    @line_cell_magic
    def cc_new(self, line: str, cell: str | None = None) -> None:
        """Start a fresh Claude Code conversation (clears session history).

        Usage::

            %cc_new Analyse this data from scratch
            %%cc_new
            Analyse this data from scratch
        """
        if cell is not None:
            line = line + "\n" + cell
        args, prompt = self._parse_args_and_prompt(line, self.cc)
        if self._handle_cc_options(args):
            return
        if not prompt:
            raise ValueError("A prompt must be provided to start the conversation.")
        self._history_manager.reset_output_tracking()
        self._variable_tracker.reset()
        self._config_manager.reset_for_new_conversation()
        self._cell_queue_manager.clear()
        if self.shell is not None:
            self.shell.user_ns.pop("_claude_cell_queue", None)
        if self._client_manager is not None:
            self._client_manager.reset_session()
        else:
            self._client_manager = ClaudeClientManager()
        self._config_manager.is_new_conversation = True
        self._execute_prompt(prompt, verbose=args.verbose)


# ---------------------------------------------------------------------------
# Extension entry point
# ---------------------------------------------------------------------------

def load_ipython_extension(ipython: object) -> None:
    """Entry point called by ``%load_ext cc_jupyter``.

    Registers ``%cc``, ``%%cc``, ``%cc_new``, and ``%ccn`` on *ipython*.
    Also wires up ``pre_run_cell`` / ``post_run_cell`` hooks for the
    :class:`CellWatcher` and :class:`ClaudeJupyterMagics`.
    """
    if not isinstance(ipython, InteractiveShell):
        return
    cell_watcher = CellWatcher()
    magics = ClaudeJupyterMagics(ipython, cell_watcher)
    ipython.register_magics(magics)
    ipython.events.register("pre_run_cell", cell_watcher.pre_run_cell)
    ipython.events.register("post_run_cell", cell_watcher.post_run_cell)
    print(HELP_TEXT)


# ---------------------------------------------------------------------------
# Self-test (run with: uv run cc_jupyter.py test)
# Interactive shell (run with: uv run cc_jupyter.py)
# ---------------------------------------------------------------------------


def _run_self_test() -> None:
    """Run the built-in smoke tests."""
    import io
    import tempfile
    import time as _time
    from contextlib import redirect_stdout
    from IPython.core.interactiveshell import InteractiveShell

    _passed = 0
    _failed = 0

    def _test(name: str, fn: Any) -> None:
        nonlocal _passed, _failed
        try:
            fn()
            print(f"  ✅ {name}")
            _passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            _failed += 1

    print("Running cc_jupyter smoke tests…\n")

    # -- CellWatcher ----------------------------------------------------------
    print("CellWatcher")

    def _t_watcher_init():
        w = CellWatcher()
        assert hasattr(w, "last_cell_finish_time")
        assert len(w.time_between_cell_executions) == 0

    def _t_watcher_not_queued():
        w = CellWatcher()
        assert not w.was_execution_probably_queued()

    def _t_watcher_queued():
        w = CellWatcher()
        # Simulate two rapid executions
        w.time_between_cell_executions.append(0.01)
        w.time_between_cell_executions.append(0.01)
        assert w.was_execution_probably_queued()

    _test("initialises without shell", _t_watcher_init)
    _test("not queued with empty history", _t_watcher_not_queued)
    _test("detects queued execution", _t_watcher_queued)

    # -- CellQueueManager -----------------------------------------------------
    print("\nCellQueueManager")

    def _t_queue_add():
        q = CellQueueManager()
        pos = q.add_cell({"code": "x=1", "executed": False, "marker_id": "a"})
        assert pos == 1
        assert len(q.get_queue()) == 1

    def _t_queue_mark_executed():
        q = CellQueueManager()
        q.add_cell({"code": "x=1", "executed": False, "marker_id": "a"})
        q.mark_executed(0)
        assert q.get_queue()[0]["executed"] is True

    def _t_queue_thread_safety():
        q = CellQueueManager()
        n = 100
        threads = [
            threading.Thread(target=lambda: [
                q.add_cell({"code": f"x={i}", "executed": False, "marker_id": str(i)})
                for i in range(n)
            ])
            for _ in range(4)
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(q.get_queue()) == 4 * n

    _test("add_cell returns position", _t_queue_add)
    _test("mark_executed sets flag", _t_queue_mark_executed)
    _test("thread-safe concurrent adds", _t_queue_thread_safety)

    # -- ConfigManager --------------------------------------------------------
    print("\nConfigManager")

    def _t_config_defaults():
        c = ConfigManager()
        assert c.model == "sonnet"
        assert c.max_cells == 3
        assert c.timeout_seconds == 300
        assert c.is_new_conversation is True

    def _t_config_reset():
        c = ConfigManager()
        c.is_new_conversation = False
        c.create_python_cell_count = 5
        c.reset_for_new_conversation()
        assert c.is_new_conversation is True
        assert c.create_python_cell_count == 0

    _test("defaults", _t_config_defaults)
    _test("reset_for_new_conversation", _t_config_reset)

    # -- prepare_imported_files_content ---------------------------------------
    print("\nprepare_imported_files_content")

    def _t_import_empty():
        assert prepare_imported_files_content([]) == ""

    def _t_import_file():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 42\n")
            name = f.name
        result = prepare_imported_files_content([name])
        assert "x = 42" in result
        Path(name).unlink()

    _test("empty list returns empty string", _t_import_empty)
    _test("reads and formats a real file", _t_import_file)

    # -- build_enhanced_prompt ------------------------------------------------
    print("\nbuild_enhanced_prompt")

    def _t_prompt_str():
        result = build_enhanced_prompt("hello", "no vars")
        assert isinstance(result, str)
        assert "hello" in result

    def _t_prompt_with_images():
        imgs = [{"format": "image/png", "data": "abc123"}]
        result = build_enhanced_prompt("hello", "no vars", captured_images=imgs)
        assert isinstance(result, list)
        assert any(b.get("type") == "image" for b in result)
        assert any(b.get("type") == "text" for b in result)

    _test("returns string without images", _t_prompt_str)
    _test("returns list with images", _t_prompt_with_images)

    # -- IPython integration --------------------------------------------------
    print("\nIPython integration")

    def _t_load_extension():
        shell = InteractiveShell.instance()
        buf = io.StringIO()
        with redirect_stdout(buf):
            load_ipython_extension(shell)
        out = buf.getvalue()
        assert "Claude Code Magic loaded" in out
        assert "%cc" in out
        InteractiveShell.clear_instance()

    def _t_hooks_registered():
        shell = InteractiveShell.instance()
        with redirect_stdout(io.StringIO()):
            load_ipython_extension(shell)
        cbs = shell.events.callbacks
        assert len(cbs.get("pre_run_cell", [])) >= 1
        assert len(cbs.get("post_run_cell", [])) >= 1
        InteractiveShell.clear_instance()

    _test("load_ipython_extension prints help", _t_load_extension)
    _test("hooks registered on shell", _t_hooks_registered)

    # -- Claude CLI availability ----------------------------------------------
    print("\nClaude CLI")

    def _t_claude_available():
        import shutil
        import subprocess
        assert shutil.which("claude") is not None, (
            "claude CLI not found in PATH — install with:\n"
            "  curl -fsSL https://claude.ai/install.sh | bash   (macOS/Linux)\n"
            "  irm https://claude.ai/install.ps1 | iex           (Windows)"
        )
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, f"claude --version failed: {result.stderr}"
        assert "Claude Code" in result.stdout, f"Unexpected output: {result.stdout!r}"

    _test("claude CLI in PATH and responds to --version", _t_claude_available)

    def _t_claude_query():
        import subprocess
        result = subprocess.run(
            ["claude", "-p", "reply with just the word ok", "--model", "haiku"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"claude query failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr.strip()}\n"
            "Check: claude auth login, or that you are not in a restricted directory."
        )
        assert result.stdout.strip(), "claude returned an empty response"

    _test("claude can run a live query in current directory", _t_claude_query)

    # -- Summary --------------------------------------------------------------
    print(f"\n{'─' * 40}")
    print(f"  {_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)


TUTORIAL_TEXT = """\
╔══════════════════════════════════════════════════════════════════════╗
║                     cc_jupyter — Quick Tutorial                     ║
╚══════════════════════════════════════════════════════════════════════╝

┌─ Debugging ──────────────────────────────────────────────────────────┐
│                                                                      │
│  %pdb on          Auto-enter debugger on any unhandled exception     │
│  %debug           Post-mortem — inspect the frame after an error     │
│  breakpoint()     Inline breakpoint in your code (Python 3.7+)      │
│  %run -d file.py  Run a script under the debugger from the start    │
│  %xmode Verbose   Show local variables in tracebacks                │
│                                                                      │
│  Tip: run  %pdb on  at session start — when Claude-generated code   │
│  fails you'll land right in the frame to inspect variables.          │
│                                                                      │
│  Recommended packages:                                               │
│    pdbpp   — drop-in pdb replacement with sticky mode & colors      │
│    pudb    — full TUI debugger (source, vars, stack, breakpoints)   │
│    ipdb    — pdb + IPython completion & highlighting                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Matplotlib (sixel output enabled) ─────────────────────────────────┐
│                                                                      │
│  Plots render inline as sixel graphics in your terminal.             │
│                                                                      │
│  import matplotlib.pyplot as plt                                     │
│  plt.plot([1, 2, 3], [1, 4, 9]); plt.title("test"); plt.show()     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ Claude Code Magics ────────────────────────────────────────────────┐
│                                                                      │
│  %cc <prompt>          Ask Claude to generate code (one-line)        │
│  %%cc                  Multi-line prompt to Claude                   │
│  %cc_new / %ccn        Start a fresh conversation                   │
│  %cc --model opus      Use a stronger model                         │
│  %cc --import f.py     Add file context                             │
│  %cc --help            All flags                                    │
│                                                                      │
│  Tip: end a %%cc multi-line block with a blank line (just Enter).   │
│  If stuck, try Esc then Enter, or Ctrl+D on the empty ... line.     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
"""

_MATPLOTLIB_SIXEL_SETUP = """\
import matplotlib, io, subprocess, shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import _Backend, FigureManagerBase
from matplotlib._pylab_helpers import Gcf

class _SixelManager(FigureManagerBase):
    def show(self):
        buf = io.BytesIO()
        self.canvas.figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        p = subprocess.Popen(["img2sixel"], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.communicate(input=buf.getvalue())

class _SixelCanvas(FigureCanvasAgg):
    manager_class = _SixelManager

@_Backend.export
class _BackendSixel(_Backend):
    FigureCanvas = _SixelCanvas
    FigureManager = _SixelManager
    mainloop = staticmethod(lambda: None)
    @classmethod
    def show(cls, *args, **kwargs):
        _Backend.show(*args, **kwargs)
        Gcf.destroy_all()

matplotlib.use("module://__main__")
import matplotlib.pyplot as plt
if shutil.which("img2sixel"):
    print("matplotlib backend: sixel (inline via img2sixel)")
else:
    print("WARNING: img2sixel not found — sudo apt install libsixel-bin")
"""


def _run_interactive_shell(*, tutorial: bool = False) -> None:
    """Launch an interactive IPython shell with cc_jupyter loaded and colors enabled."""
    from IPython import start_ipython

    startup_code = _MATPLOTLIB_SIXEL_SETUP
    if tutorial:
        # Print tutorial before the matplotlib setup
        startup_code = f"print({TUTORIAL_TEXT!r})\n" + startup_code

    start_ipython(
        argv=[
            "--colors=Linux",
            "--ext=cc_jupyter",
            "--TerminalInteractiveShell.term_title=False",
            "--TerminalInteractiveShell.prompts_class=IPython.terminal.prompts.ClassicPrompts",
            "-c", startup_code,
            "-i",
        ],
    )


if __name__ == "__main__":
    import dataclasses

    import tyro

    @dataclasses.dataclass
    class Args:
        """cc_jupyter — Claude Code magic for IPython/Jupyter."""

        test: bool = False
        """Run the built-in smoke tests instead of launching the interactive shell."""

        tutorial: bool = False
        """Show the quick tutorial on startup."""

    args = tyro.cli(Args)
    if args.test:
        _run_self_test()
    else:
        _run_interactive_shell(tutorial=args.tutorial)
