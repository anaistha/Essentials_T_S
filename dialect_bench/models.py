import os

import requests

THINKING = {"on": True, "off": False, "default": None}


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
        """Returns the answer and the model's reasoning (None if it didn't think).

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
        message = r.json()["message"]
        return message["content"], message.get("thinking") or None
