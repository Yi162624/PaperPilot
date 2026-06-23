"""
AI 模型客户端
封装 Kimi 文本模型 API 调用
"""

import openai
import base64
from pathlib import Path
from typing import Optional, List, Dict

from config import get_settings

settings = get_settings()


class KimiClient:
    """
    Kimi 多模态模型客户端
    使用 Moonshot AI 的 OpenAI 兼容接口，支持图文混传
    """

    def __init__(self, model: Optional[str] = None):
        self.client = openai.OpenAI(         # 初始化 OpenAI 客户端
            api_key=settings.kimi_api_key,
            base_url=settings.kimi_base_url
        )
        self.model = model or settings.kimi_model

    def _encode_image(self, img_path: str) -> str:
        """把图片文件编码成 base64 的 data URL"""
        ext = Path(img_path).suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{ext};base64,{img_data}"

    def chat_completion(
        self,
        prompt: str,
        messages: Optional[List[Dict]] = None,
        max_tokens: int = 8000,
        temperature: float = 1.0,  # Kimi k2.6 要求 temperature=1.0
        images: Optional[List[str]] = None  # 图片文件路径列表
    ) -> str:
        """调用 Kimi 进行对话补全，有图片时自动切换图文混传模式"""
        if messages is None:
            messages = []

        # 有图片 → 多模态 content 数组；没图片 → 纯字符串
        if images:
            content = []
            for img_path in images:
                try:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(img_path)}
                    })
                except Exception:
                    pass
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            # 调用 Kimi API 进行对话补全
            response = self.client.chat.completions.create(
                model=self.model,             # 模型名称
                messages=messages,            # 输入消息列表
                max_tokens=max_tokens,        # 最大生成 token 数
                temperature=temperature,      # 温度参数
                stream=False
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            return "未获取到有效回复"
        except Exception as e:
            raise Exception(f"调用 Kimi API 失败: {str(e)}")