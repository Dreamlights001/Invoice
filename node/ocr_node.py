"""OCR 节点，以及配置、状态和模型调用工具。"""
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from typing_extensions import TypedDict


_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_OCR_SUFFIXES = (
    ".deepseek_ocr_raw.txt",
    ".deepseek_ocr_raw",
    ".ocr.txt",
    ".invoice.json",
    ".truth.json",
    ".json",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)

DEFAULT_OCR_PROMPT = "<image>\n<|grounding|> Convert the document to text layout preserving format."

_OCR_ATOM_PATTERN = re.compile(
    r"<\|ref\|>(?P<kind>.*?)<\|/ref\|>"
    r"<\|det\|>\[\[(?P<coords>.*?)\]\]<\|/det\|>\s*"
    r"(?P<content>.*?)(?=<\|ref\|>|$)",
    re.DOTALL,
)


@dataclass
class LLMConfig:
    """大模型调用配置。"""

    api_url: str
    api_key: str
    model: str


@dataclass
class OCRConfig:
    """OCR 服务配置。"""

    base_url: str
    api_key: str
    model: str


@dataclass
class EmbeddingConfig:
    """向量检索配置。"""

    url: str
    api_key: str
    model: str


@dataclass
class RerankConfig:
    """重排模型配置。"""

    url: str
    api_key: str
    model: str


@dataclass
class Config:

    llm: LLMConfig
    reflect_llm: LLMConfig
    ocr: OCRConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    rag_db_path: Optional[Path]
    fewshot_dir: Optional[Path]
    base_dir: Path


class InvoiceState(TypedDict):
    """发票处理过程中的共享状态。"""

    input: Path
    doc_id: str
    method: str
    ocr_raw_text: Optional[str]
    ocr_atoms: Optional[List[Dict[str, Any]]]
    blocks: Optional[Dict[str, Dict[str, str]]]
    rag_doc_examples: Optional[List[Dict[str, Any]]]
    block_parses: Optional[Dict[str, Dict[str, Any]]]
    merged_invoice: Optional[Dict[str, Any]]
    reflected_invoice: Optional[Dict[str, Any]]
    final_invoice: Optional[Dict[str, Any]]
    truth: Optional[Dict[str, Any]]
    evaluation_result: Optional[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]
    prompts: Dict[str, Any]


@dataclass
class OcrAtom:
    """单个 OCR 原子块。"""

    type: str
    text: str
    box_2d: List[float]


class OCRCache:
    """负责 OCR 文本缓存读写。"""

    def __init__(self, root: Path):
        self.root = root

    def locate(self, doc_id: str) -> Optional[Path]:
        """按文档编号定位缓存文件。"""
        if not self.root.exists():
            return None

        for exact_name in (f"{doc_id}.deepseek_ocr_raw.txt", f"{doc_id}.deepseek_ocr_raw"):
            candidate = self.root / exact_name
            if candidate.is_file():
                return candidate

        for path in self.root.iterdir():
            if path.is_file() and "deepseek_ocr_raw" in path.name and normalize_name(path.name) == doc_id:
                return path
        return None

    def write(self, doc_id: str, raw_text: str) -> Path:
        """将 OCR 文本写入缓存目录。"""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{doc_id}.deepseek_ocr_raw.txt"
        target.write_text(raw_text, encoding="utf-8")
        return target


class LLMService:
    """基于 OpenAI SDK 的调用封装。"""

    def __init__(self, config: LLMConfig):
        self._client = OpenAI(api_key=config.api_key, base_url=config.api_url)
        self._model = config.model

    def invoke(self, prompt: str, parse_json: bool = True) -> Any:
        """发起单轮对话调用，并在需要时尝试提取 JSON。"""
        raw_content = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content or ""
        if not parse_json:
            return raw_content
        parsed = self.extract_json(raw_content)
        return raw_content if parsed is None else parsed

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """从模型输出中提取字典对象。"""
        source = (text or "").strip()
        if not source:
            return None

        cleaned = re.sub(r"^```(?:json)?\s*", "", source)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        for candidate in (cleaned, source, _slice_first_json_object(cleaned), _slice_first_json_object(source)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None


def load_config(config_path: Path) -> Config:
    """从本地 JSON 文件加载运行配置。"""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    rag_db_path = _default_rag_db_path(base_dir)
    fewshot_dir = _default_fewshot_dir(base_dir)

    llm_config = LLMConfig(
        api_url=str(payload.get("LLM_API_URL", "") or ""),
        api_key=str(payload.get("LLM_API_KEY", "") or ""),
        model=str(payload.get("LLM_MODEL", "") or ""),
    )
    reflect_llm_config = LLMConfig(
        api_url=str(payload.get("reflect_LLM_API_URL", "") or llm_config.api_url or ""),
        api_key=str(payload.get("reflect_LLM_API_KEY", "") or llm_config.api_key or ""),
        model=str(payload.get("reflect_LLM_MODEL", "") or llm_config.model or ""),
    )

    return Config(
        llm=llm_config,
        reflect_llm=reflect_llm_config,
        ocr=OCRConfig(
            base_url=str(payload.get("ocr_base_url", "") or ""),
            api_key=str(payload.get("ocr_api_key", "") or ""),
            model=str(payload.get("ocr_model", "") or ""),
        ),
        embedding=EmbeddingConfig(
            url=str(payload.get("EMBEDDING_URL", "") or ""),
            api_key=str(payload.get("EMBEDDING_API_KEY", "") or ""),
            model=str(payload.get("EMBEDDING_MODEL", "") or ""),
        ),
        rerank=RerankConfig(
            url=str(payload.get("rerank_url", "") or ""),
            api_key=str(payload.get("rerank_api", "") or ""),
            model=str(payload.get("rerank_model", "") or ""),
        ),
        rag_db_path=rag_db_path,
        fewshot_dir=fewshot_dir,
        base_dir=base_dir,
    )


def normalize_name(value: str) -> str:
    """将文件名或标识转换为稳定的匹配编号。"""
    original_name = Path(value).name
    lowered_name = original_name.lower()

    stripped = original_name
    for suffix in _OCR_SUFFIXES:
        if lowered_name.endswith(suffix):
            stripped = original_name[: len(original_name) - len(suffix)]
            break

    return stripped.replace(" ", "_").replace(".", "_").replace("-", "_")


def normalize_doc_id(input_path: Path) -> str:
    """将输入路径转换为流程内部使用的文档编号。"""
    return normalize_name(input_path.name)


def build_initial_state(input: Path, truth: Optional[Dict[str, Any]] = None, method: str = "base") -> InvoiceState:
    """创建单据处理的初始共享状态。"""
    return {
        "input": input,
        "doc_id": normalize_doc_id(input),
        "method": method,
        "ocr_raw_text": None,
        "ocr_atoms": None,
        "blocks": None,
        "rag_doc_examples": None,
        "block_parses": None,
        "merged_invoice": None,
        "reflected_invoice": None,
        "final_invoice": None,
        "truth": truth,
        "evaluation_result": None,
        "errors": [],
        "metadata": {"method": method},
        "prompts": {},
    }


def parse_deepseek_ocr_atoms(raw_text: str) -> List[OcrAtom]:
    """解析 DeepSeek OCR 文本中的定位片段。"""
    atoms: List[OcrAtom] = []
    for match in _OCR_ATOM_PATTERN.finditer(raw_text or ""):
        box = _parse_box(match.group("coords") or "")
        if box is None:
            continue
        atoms.append(
            OcrAtom(
                type=(match.group("kind") or "").strip(),
                text=(match.group("content") or "").strip(),
                box_2d=box,
            )
        )
    return atoms


def call_deepseek_ocr(file_path: Path, config: OCRConfig, prompt: str = DEFAULT_OCR_PROMPT) -> str:
    """调用 OCR 模型，将文件转换成带版面信息的文本。"""
    data_url = _encode_as_data_url(file_path)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def ocr_node(state: InvoiceState, config: Config, intervar: Optional[Path] = None) -> InvoiceState:
    """执行 OCR 阶段，默认直连接口；仅在显式提供 intervar 时启用缓存。"""
    doc_id = state["doc_id"]

    try:
        raw_text: str
        cache_path: Optional[Path] = None
        source = "api"
        cache_enabled = intervar is not None

        if cache_enabled:
            cache = OCRCache(intervar)
            cache_path = cache.locate(doc_id)
            if cache_path is not None:
                raw_text = cache_path.read_text(encoding="utf-8")
                source = "interVar"
            else:
                raw_text = call_deepseek_ocr(state["input"], config.ocr)
                cache_path = cache.write(doc_id, raw_text)
        else:
            raw_text = call_deepseek_ocr(state["input"], config.ocr)

        atoms = parse_deepseek_ocr_atoms(raw_text)
        state["ocr_raw_text"] = raw_text
        state["ocr_atoms"] = [_atom_to_dict(atom) for atom in atoms]
        state["metadata"].update(
            {
                "ocr_source": source,
                "ocr_cache_enabled": cache_enabled,
                "ocr_cache_hit": source == "interVar",
                "ocr_cache_path": str(cache_path) if cache_path is not None else "",
                "ocr_atom_count": len(atoms),
            }
        )
    except Exception as exc:
        state["ocr_raw_text"] = None
        state["ocr_atoms"] = None
        state["errors"].append(f"OCR failed: {exc}")
    return state


def _slice_first_json_object(text: str) -> str:
    """截取文本中的首个 JSON 对象片段。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def _encode_as_data_url(file_path: Path) -> str:
    """将文件编码为模型接口可直接使用的 data url。"""
    suffix = file_path.suffix.lower()
    mime = _MIME_TYPES.get(suffix)
    if mime is None:
        raise ValueError(f"不支持的文件类型: {suffix}")
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _parse_box(text: str) -> Optional[List[float]]:
    """解析 OCR 返回的四点坐标。"""
    try:
        values = [float(item.strip()) for item in text.split(",")]
    except Exception:
        return None
    return values if len(values) == 4 else None


def _atom_to_dict(atom: OcrAtom) -> Dict[str, Any]:
    """将 OCR 原子对象转为状态中可序列化的字典。"""
    return {"type": atom.type, "text": atom.text, "box_2d": atom.box_2d}


def _default_rag_db_path(base_dir: Path) -> Optional[Path]:
    """返回项目内固定的知识库路径。"""
    candidate = (base_dir / "knowledge" / "knowledge.sqlite").resolve()
    return candidate if candidate.exists() else None


def _default_fewshot_dir(base_dir: Path) -> Optional[Path]:
    """返回项目内固定的 few-shot 目录。"""
    candidate = (base_dir / "fewshot_precomputed").resolve()
    return candidate if candidate.exists() else None
