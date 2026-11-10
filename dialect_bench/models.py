import os
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests

THINKING = {"on": True, "off": False, "default": None}


class CodexModel:
    """One fresh, read-only Codex CLI invocation per question."""

    def __init__(self, tag):
        self.tag = tag
        self.can_think = False  # The Luna benchmark condition has reasoning effort none.
        self.options = {"sandbox": "read-only", "ephemeral": True, "parallel": 1}
        self.executable = shutil.which("codex")
        if not self.executable:
            raise RuntimeError("codex was not found; install the CLI and open a new terminal")
        version = subprocess.run([self.executable, "--version"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", check=True)
        self.version = version.stdout.strip()

    def describe(self):
        return {"tag": self.tag, "codex_version": self.version, "provider": "chatgpt-cli"}

    def ask(self, prompt, thinking=None):
        if thinking is True:
            raise ValueError("This Codex benchmark configuration does not support thinking=on")
        # Only the prompt enters the isolated working directory. The agent cannot
        # look up German originals or ground-truth answers in the benchmark repo.
        with tempfile.TemporaryDirectory(prefix="dialect_codex_") as workspace:
            output = Path(workspace) / "answer.txt"
            command = [self.executable, "exec", "--model", self.tag,
                       "--sandbox", "read-only", "--ephemeral",
                       "--skip-git-repo-check", "--ignore-user-config",
                       "--json", "--output-last-message", str(output), prompt]
            result = subprocess.run(command, cwd=workspace, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=600)
            if result.returncode:
                raise RuntimeError(f"codex exited {result.returncode}: {result.stderr[-500:]}")
            events = []
            for line in result.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if any(event.get("type") == "turn.failed" for event in events):
                raise RuntimeError("codex reported turn.failed")
            if not output.is_file():
                raise RuntimeError(f"codex did not write a final answer: {result.stderr[-500:]}")
            answer = output.read_text(encoding="utf-8-sig").strip()
            if not answer:
                raise RuntimeError("codex returned an empty final answer")
            # CLI reasoning summaries are not comparable to Ollama's thinking field.
            return answer, None, "stop"


class ClaudeCodeModel:
    """One fresh Claude Code CLI call (claude -p) per question, using the signed-in Claude subscription.

    The call runs in an empty folder with no tools, no MCP servers, no settings files and an empty
    system prompt, so the model only sees the question.
    """

    def __init__(self, tag, effort="high"):
        self.tag = tag
        self.can_think = True
        self.options = {"effort": effort, "tools": "none", "system_prompt": ""}
        self.executable = shutil.which("claude")
        if not self.executable:
            raise RuntimeError("claude was not found; install Claude Code and sign in with `claude`")
        version = subprocess.run([self.executable, "--version"], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", check=True)
        self.version = version.stdout.strip()

    def describe(self):
        return {"tag": self.tag, "claude_code_version": self.version, "provider": "claude-code"}

    def ask(self, prompt, thinking=None):
        command = [self.executable, "-p", "--model", self.tag, "--effort", self.options["effort"],
                   "--output-format", "json", "--tools", "", "--strict-mcp-config",
                   "--setting-sources", "", "--no-session-persistence", "--system-prompt", ""]
        # Tested with Claude Code 2.1.322: with alwaysThinkingEnabled=false, Haiku 5.5 still thinks on every
        # question and Sonnet 5.5 on some, so thinking can't really be turned off through the CLI
        if thinking is False:
            raise ValueError("thinking=off is not supported for Claude through Claude Code")

        while True:
            with tempfile.TemporaryDirectory(prefix="dialect_claude_") as workspace:
                result = subprocess.run(command, input=prompt, cwd=workspace, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=3600)
            try:
                out = json.loads(result.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(f"claude exited {result.returncode}: {result.stderr[-500:]}")

            message = str(out.get("result", ""))
            if out.get("is_error"):
                # out of usage for now: wait for the limit to reset instead of failing every question
                if out.get("api_error_status") == 429 or "limit" in message.lower():
                    print(f"claude usage limit reached ({message[:200]}), waiting 15 minutes")
                    time.sleep(15 * 60)
                    continue
                raise RuntimeError(f"claude returned an error: {message[:500]}")
            break

        used = set(out.get("modelUsage", {}))
        if used and self.tag not in used:
            raise RuntimeError(f"asked for {self.tag}, but claude used {sorted(used)}")
        answer = message.strip()
        if not answer:
            raise RuntimeError("claude returned an empty answer")
        # Claude never returns its raw reasoning, so there is nothing for the reasoning column
        return answer, None, "length" if out.get("stop_reason") == "max_tokens" else "stop"


class OllamaModel:
    def __init__(self, tag, options):
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        self.url = host if host.startswith("http") else "http://" + host
        self.tag = tag
        self.options = options

        info = self.show()
        if info is None:
            print(f"pulling {tag} ...")
            r = requests.post(f"{self.url}/api/pull", json={"model": tag, "stream": False}, timeout=None)
            r.raise_for_status()
            info = self.show()
        self.info = info
        self.can_think = "thinking" in info.get("capabilities", [])

    @property
    def is_cloud(self):
        return self.tag.endswith("-cloud")

    def show(self):
        r = requests.post(f"{self.url}/api/show", json={"model": self.tag}, timeout=60)
        return r.json() if r.ok else None

    def fits_on_gpu(self):
        """Loads the model and checks that all of it is in GPU memory.

        Ollama reserves memory for every parallel request (OLLAMA_NUM_PARALLEL). If that is too much,
        it puts part of the model on the CPU instead, which makes it very slow.
        """
        if self.is_cloud:
            print(f"{self.tag}: cloud model, skipping local GPU check")
            return True

        r = requests.post(f"{self.url}/api/generate", json={"model": self.tag, "options": self.options}, timeout=1800)
        r.raise_for_status()
        loaded = requests.get(f"{self.url}/api/ps", timeout=60).json()["models"]
        model = next(m for m in loaded if self.tag in (m["name"], m["model"]))
        print(f"{self.tag}: {model['size_vram'] / 1e9:.1f} of {model['size'] / 1e9:.1f} GB on the GPU")
        return model["size_vram"] >= model["size"]

    def describe(self):
        """The model's size and quantization and the Ollama version, for run_info.jsonl."""
        details = self.info.get("details", {})
        return {
            "tag": self.tag,
            "size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "ollama_version": requests.get(f"{self.url}/api/version", timeout=60).json().get("version"),
        }

    def ask(self, prompt, thinking=None):
        """Returns the answer, the model's reasoning (None if it didn't think) and why the model stopped:
        "stop" if it finished, "length" if it hit num_predict.

        thinking=None means we let the model decide whether to think.
        """
        body = {
            "model": self.tag,
            "messages": [{"role": "user", "content": prompt}],
            "options": self.options,
            "stream": False,
        }
        # Ollama gives an error if we send "think" to a model that can't think
        if thinking is not None and self.can_think:
            body["think"] = thinking
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=3600)
        if not r.ok:
            raise RuntimeError(f"ollama returned {r.status_code}: {r.text[:300]}")
        result = r.json()
        message = result["message"]
        return message["content"], message.get("thinking") or None, result.get("done_reason")
