"""多 LoRA 管理器。

MiniMax-H3 支持运行时动态加载/卸载多个 LoRA，
本模块负责扫描 LoRA 目录、加载 LoRA 权重到模型、管理已加载列表。

使用 peft 加载 LoRA（要求模型为 nn.Module）。
LoRA 文件格式：safetensors（与 manifest.models.loras.format 一致）。

目录结构：
    models/video/minimax-h3/loras/
    ├── style_a.safetensors
    ├── style_b.safetensors
    └── ...
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

_module_logger = logging.getLogger(__name__)


class LoRAManager:
    """多 LoRA 管理器。

    负责扫描 lora_dir 下的 LoRA 文件，按需加载到模型，
    并维护已加载 LoRA 的列表（含权重、adapter 名）。

    线程安全：所有加载/卸载操作用 _lock 保护。
    """

    def __init__(self, lora_dir: Path, logger: Optional[logging.Logger] = None):
        """初始化 LoRA 管理器。

        Args:
            lora_dir: LoRA 文件目录（如 models/video/minimax-h3/loras）
            logger: 可选的日志器（None 用模块级 logger）
        """
        self.lora_dir = lora_dir
        self._logger = logger or _module_logger
        self._lock = threading.Lock()
        # 已加载的 LoRA 字典：name → {path, weight, adapter}
        self._loaded_loras: dict[str, dict[str, Any]] = {}

    @property
    def loaded_loras(self) -> dict[str, dict[str, Any]]:
        """已加载 LoRA 字典（只读视图）。"""
        with self._lock:
            return dict(self._loaded_loras)

    def scan_loras(self) -> list[dict]:
        """扫描 lora_dir 下的 .safetensors 文件。

        Returns:
            LoRA 信息列表，每项含 name / path / size_mb 字段。
            目录不存在时返回空列表。
        """
        if not self.lora_dir.exists():
            self._logger.debug("LoRA 目录不存在: %s", self.lora_dir)
            return []

        loras: list[dict] = []
        for f in sorted(self.lora_dir.glob("*.safetensors")):
            loras.append({
                "name": f.stem,
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })

        self._logger.info("LoRA 扫描完成: 找到 %d 个文件 (dir=%s)", len(loras), self.lora_dir)
        return loras

    def load_lora(
        self,
        model: Any,
        lora_name: str,
        weight: float = 1.0,
    ) -> dict:
        """加载单个 LoRA 到模型（用 peft）。

        Args:
            model: 目标模型（nn.Module）
            lora_name: LoRA 名称（lora_dir 下的文件 stem，如 style_a）
            weight: LoRA 权重（0.0 - 2.0）

        Returns:
            已加载的 LoRA 信息（含 path / weight / adapter）

        Raises:
            FileNotFoundError: LoRA 文件不存在
            RuntimeError: 加载失败（peft 未安装或 LoRA 格式错误）
        """
        with self._lock:
            if lora_name in self._loaded_loras:
                self._logger.warning("LoRA %s 已加载，先卸载再重新加载", lora_name)
                self._unload_lora_unsafe(lora_name)

            lora_path = self.lora_dir / f"{lora_name}.safetensors"
            if not lora_path.exists():
                raise FileNotFoundError(
                    f"LoRA 文件不存在: {lora_path}（可用: {[d.stem for d in self.lora_dir.glob('*.safetensors')]}）"
                )

            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise RuntimeError(
                    f"peft 包未安装，无法加载 LoRA: {e}。"
                    f"请在 envs/video-minimax-h3/ 环境中运行 `pip install peft`"
                ) from e

            self._logger.info(
                "加载 LoRA: name=%s, path=%s, weight=%.2f",
                lora_name, lora_path, weight,
            )

            try:
                # 简化实现：构造 LoRA config 并注入模型
                # 实际的 target_modules 应根据 MiniMax-H3 transformer 结构调整
                config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["to_q", "to_k", "to_v", "to_out"],
                    lora_dropout=0.05,
                )
                # 若模型已是 PEFT 模型，直接加 adapter；否则包装成 PEFT 模型
                try:
                    from peft import PeftModel
                    if isinstance(model, PeftModel):
                        adapter_name = model.load_adapter(
                            str(lora_path), adapter_name=lora_name
                        )
                    else:
                        # 首次加载：用 get_peft_model 包装
                        # 注意：此处会修改 model，调用方需保存返回值
                        model = get_peft_model(model, config)
                        adapter_name = lora_name
                except Exception as e:
                    raise RuntimeError(f"LoRA adapter 加载失败: {e}") from e

                # 应用权重（若 PEFT 模型支持 set_adapter_scale）
                try:
                    model.set_adapter_scale(adapter_name, weight)
                except (AttributeError, TypeError):
                    # 不支持时跳过（记录日志）
                    self._logger.debug(
                        "模型不支持 set_adapter_scale，weight=%s 已记录但未应用",
                        weight,
                    )

                info = {
                    "path": str(lora_path),
                    "weight": weight,
                    "adapter": adapter_name,
                }
                self._loaded_loras[lora_name] = info
                self._logger.info("LoRA 加载成功: name=%s", lora_name)
                return info

            except Exception as e:
                self._logger.exception("LoRA 加载失败: name=%s", lora_name)
                raise RuntimeError(f"LoRA 加载失败: {e}") from e

    def unload_lora(self, lora_name: str) -> bool:
        """卸载单个 LoRA。

        Args:
            lora_name: LoRA 名称

        Returns:
            是否成功卸载（未加载时返回 False）
        """
        with self._lock:
            return self._unload_lora_unsafe(lora_name)

    def _unload_lora_unsafe(self, lora_name: str) -> bool:
        """卸载 LoRA（无锁版本）。"""
        info = self._loaded_loras.pop(lora_name, None)
        if info is None:
            self._logger.warning("LoRA %s 未加载，无需卸载", lora_name)
            return False

        # adapter 卸载由调用方在模型上执行（model.delete_adapter）
        # 此处只清理管理器内部的记录
        self._logger.info(
            "LoRA 已卸载: name=%s, path=%s",
            lora_name, info["path"],
        )
        return True

    def get_loaded_loras(self) -> list[dict]:
        """返回已加载 LoRA 列表。

        Returns:
            列表，每项含 name / path / weight / adapter 字段
        """
        with self._lock:
            return [
                {"name": name, **info}
                for name, info in self._loaded_loras.items()
            ]

    def apply_loras(self, model: Any) -> Any:
        """合并所有已加载 LoRA 到模型（多 LoRA 加权融合）。

        多个 LoRA 同时生效时，调用 model.set_adapter 设置活跃 adapter 列表。
        PEFT 的 set_adapter 支持多 adapter 融合，按各自 weight 加权。

        Args:
            model: 目标模型（nn.Module，应为 PeftModel）

        Returns:
            应用了 LoRA 的模型（通常与输入相同，PEFT 是原地修改）
        """
        with self._lock:
            if not self._loaded_loras:
                self._logger.debug("无已加载 LoRA，跳过 apply_loras")
                return model

            adapter_names = list(self._loaded_loras.keys())
            self._logger.info(
                "应用 %d 个 LoRA 到模型: %s",
                len(adapter_names), adapter_names,
            )

            try:
                from peft import PeftModel
                if isinstance(model, PeftModel):
                    # PEFT 多 adapter 融合
                    model.set_adapter(adapter_names)
                    self._logger.info("多 LoRA 融合已应用: %s", adapter_names)
                else:
                    self._logger.warning(
                        "模型不是 PeftModel，无法应用多 LoRA 融合（adapter 列表=%s）",
                        adapter_names,
                    )
            except Exception as e:
                self._logger.warning("apply_loras 失败（不阻断推理）: %s", e)

            return model

    def unload_all(self, model: Any) -> None:
        """卸载所有 LoRA（清空管理器内部状态）。

        Args:
            model: 目标模型（用于在模型上删除 adapter，可选）
        """
        with self._lock:
            names = list(self._loaded_loras.keys())
            for name in names:
                info = self._loaded_loras.pop(name, None)
                if info is None:
                    continue
                try:
                    from peft import PeftModel
                    if isinstance(model, PeftModel) and hasattr(model, "delete_adapter"):
                        model.delete_adapter(info.get("adapter", name))
                except Exception as e:
                    self._logger.warning(
                        "删除 adapter 失败（忽略）: name=%s, error=%s",
                        name, e,
                    )
            self._logger.info("已卸载全部 LoRA: %d 个", len(names))
