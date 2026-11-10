"""
网络文件下载模块 (WebGetter.py)

这个模块提供了一个简单的文件下载工具类，类似于Linux的wget命令。
主要功能：
1. 从URL下载文件
2. 显示下载进度条
3. 自动创建输出目录
4. 支持自定义文件名

使用场景：
- 下载字体文件（如SimHei.ttf）
- 下载数据集文件
- 下载其他资源文件

主要类：
- Wget: 文件下载工具类
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# 默认超时（连接, 读取）；单文件体积上限（防磁盘占满）
DEFAULT_TIMEOUT = (10, 60)
DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB


class Wget:
    """
    文件下载工具类（类似Linux的wget命令）
    """

    def __init__(
        self,
        url: str,
        output_dir: str = ".",
        filename: Optional[str] = None,
        *,
        timeout=DEFAULT_TIMEOUT,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        self.url = url
        self.output_dir = output_dir
        self.filename = filename or self._get_filename_from_url()
        self.timeout = timeout
        self.max_bytes = max_bytes

    def _get_filename_from_url(self) -> str:
        parsed = urlparse(self.url)
        return os.path.basename(parsed.path) or "downloaded_file"

    def _create_output_dir(self) -> Path:
        out_dir = Path(self.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _safe_output_path(self) -> Path:
        """仅允许输出目录内的基名文件，阻止 ../ 穿越。"""
        out_dir = self._create_output_dir()
        # 强制只取最后一段文件名，丢掉任何目录成分
        safe_name = Path(self.filename).name
        if safe_name in {"", ".", ".."}:
            raise ValueError(f"非法文件名: {self.filename!r}")
        dest = (out_dir / safe_name).resolve()
        try:
            dest.relative_to(out_dir)
        except ValueError as exc:
            raise ValueError(
                f"输出路径越界: {dest} 不在 {out_dir} 内（filename={self.filename!r}）"
            ) from exc
        return dest

    def download(self) -> bool:
        """
        执行文件下载。

        返回:
            True: 下载成功
            False: 下载失败
        """
        try:
            response = requests.head(
                self.url, timeout=self.timeout, allow_redirects=True
            )
            total_size = int(response.headers.get("content-length", 0) or 0)
            if total_size and total_size > self.max_bytes:
                print(
                    f"下载错误: content-length {total_size} 超过上限 {self.max_bytes}"
                )
                return False

            output_path = self._safe_output_path()

            response = requests.get(self.url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            written = 0
            with open(output_path, "wb") as f, tqdm(
                desc=Path(output_path).name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    if not data:
                        continue
                    written += len(data)
                    if written > self.max_bytes:
                        raise ValueError(
                            f"下载超过上限 {self.max_bytes} bytes，已中止"
                        )
                    size = f.write(data)
                    pbar.update(size)

            print(f"\n下载完成: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"下载错误: {str(e)}")
            return False
        except Exception as e:
            print(f"发生错误: {str(e)}")
            return False
