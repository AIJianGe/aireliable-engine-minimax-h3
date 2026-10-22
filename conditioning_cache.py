"""MiniMax-H3 conditioning 磁盘缓存（v0.7.7+ T2.2）。

原理参考 ComfyUI-MiniMaxH3-CLIPCached（MIT）——相同的 prompt + 参考输入得到
的 conditioning（Qwen3VL prompt_embeds，形状 (1, n_tokens, 5120)）无需每次
重新 TE 编码；磁盘缓存命中时直接加载，跳过 19.6GB TE 的编码（L2 下还跳过
TE 临时上 GPU 的 41GB 峰值），显著降低耗时与显存占用。

本模块为独立实现（非拷贝代码）：以「模型 variant + 模式 + prompt + 参考文件
指纹」为 key，把 `get_qwen3vl_prompt_embeds` 的输出保存为 `.pt` 文件，LRU
容量管理（按 mtime 淘汰最旧条目）。

用法：
    cache = ConditioningCache(cache_dir, logger=logger)
    embeds = cache.get(key)
    if embeds is None:
        embeds = encode(...)          # 正常 TE 编码
        cache.put(key, embeds)        # 编码结果写盘
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import torch


class ConditioningCache:
    """MiniMax-H3 conditioning 磁盘缓存（Qwen3VL prompt_embeds）。

    Attributes:
        cache_dir: 缓存目录（随模型 variant 目录，跨重启复用）。
        max_entries: LRU 容量上限（超过时删除 mtime 最旧的条目）。
    """

    def __init__(
        self,
        cache_dir: Path,
        max_entries: int = 32,
        logger: Optional[logging.Logger] = None,
    ):
        self._cache_dir = Path(cache_dir)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # 缓存目录创建失败不致命（缓存退化为未命中）
            logger = logger or logging.getLogger(__name__)
            logger.warning("[cond-cache] 缓存目录创建失败（缓存禁用）: %s: %s", self._cache_dir, e)
        self._max_entries = max(1, int(max_entries))
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._logger.info(
            "[cond-cache] 初始化: dir=%s, max_entries=%d",
            self._cache_dir, self._max_entries,
        )

    @staticmethod
    def make_key(*components: Any) -> str:
        """由任意组件生成稳定缓存 key（sha256 hex）。

        components 需为可 str() 的标量（字符串/数字/Path）。张量等不可直接
        str 序列化的对象必须由调用方先转为指纹（如 路径+大小+mtime）。
        """
        parts = []
        for c in components:
            if isinstance(c, bytes):
                parts.append(c.hex())
            else:
                parts.append(str(c))
        raw = "\x1f".join(parts).encode("utf-8", errors="replace")
        return hashlib.sha256(raw).hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.pt"

    def get(
        self,
        key: str,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """读取缓存。

        命中返回 `(1, n_tokens, hidden_dim)` 张量（已按 device/dtype 转换），
        未命中或损坏返回 None。

        Args:
            key: 缓存 key（ConditioningCache.make_key 生成）
            device: 目标设备（None 保持 CPU）
            dtype: 目标 dtype（None 保持原 dtype）
        """
        entry = self._entry_path(key)
        if not entry.exists():
            return None
        try:
            data = torch.load(entry, map_location="cpu", weights_only=False)
            embeds = data.get("embeds") if isinstance(data, dict) else data
            if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
                self._logger.warning("[cond-cache] 缓存内容非法，忽略: %s", entry.name)
                return None
            if isinstance(data, dict):
                meta = data.get("meta", {})
                if tuple(meta.get("shape", [])) != tuple(embeds.shape):
                    self._logger.warning("[cond-cache] 缓存 shape 不匹配，忽略: %s", entry.name)
                    return None
                if meta.get("dtype") != str(embeds.dtype).replace("torch.", ""):
                    self._logger.warning("[cond-cache] 缓存 dtype 不匹配，忽略: %s", entry.name)
                    return None
            self._logger.info(
                "[cond-cache] 命中: %s（shape=%s）", entry.name, tuple(embeds.shape),
            )
            if device is not None or dtype is not None:
                embeds = embeds.to(device=device, dtype=dtype)
            return embeds
        except Exception as e:
            self._logger.warning("[cond-cache] 读取失败（视为未命中）: %s: %s", entry.name, e)
            return None

    def put(self, key: str, embeds: torch.Tensor) -> bool:
        """写入缓存（原子写：tmp + rename）。embeds 内部会先转到 CPU。

        仅接受 3 维 `(1, n_tokens, hidden_dim)` 张量，非法输入返回 False。
        """
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
            return False
        entry = self._entry_path(key)
        try:
            with self._lock:
                cpu_embeds = embeds.detach().to("cpu")
                meta = {
                    "shape": list(cpu_embeds.shape),
                    "dtype": str(cpu_embeds.dtype).replace("torch.", ""),
                }
                tmp = entry.with_suffix(".pt.tmp")
                torch.save({"embeds": cpu_embeds, "meta": meta}, tmp)
                tmp.replace(entry)
            self._trim()
            self._logger.info(
                "[cond-cache] 写入: %s（shape=%s, dtype=%s）",
                entry.name, meta["shape"], meta["dtype"],
            )
            return True
        except Exception as e:
            self._logger.warning("[cond-cache] 写入失败: %s: %s", entry.name, e)
            return False

    def _trim(self) -> None:
        """LRU 容量管理：超过 max_entries 时删除 mtime 最旧的条目。"""
        try:
            entries = [p for p in self._cache_dir.glob("*.pt") if p.is_file()]
            if len(entries) <= self._max_entries:
                return
            entries.sort(key=lambda p: p.stat().st_mtime)
            for old in entries[: len(entries) - self._max_entries]:
                try:
                    old.unlink()
                    self._logger.info("[cond-cache] LRU 淘汰: %s", old.name)
                except OSError:
                    pass
        except Exception as e:
            self._logger.debug("[cond-cache] 容量清理失败: %s", e)

    def clear(self) -> None:
        """清空缓存目录（用户显式清理时调用；unload 不清空，保留跨重启复用）。"""
        try:
            count = 0
            for p in self._cache_dir.glob("*.pt"):
                try:
                    p.unlink()
                    count += 1
                except OSError:
                    pass
            if count:
                self._logger.info("[cond-cache] 已清空 %d 个条目", count)
        except Exception as e:
            self._logger.warning("[cond-cache] 清空失败: %s", e)
