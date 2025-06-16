"""DO NOT MODIFY THIS FILE.

此文件包含ImageLoader类，用于下载和缓存图像。评估期间，缓存已预加载所有可用图像，
所有操作均从缓存读取。评估环境禁用网络访问，使用非数据集提供的URL会导致错误（所有`requests.get`调用会失败）。
"""

from hashlib import sha256
from io import BytesIO
import os
import requests
from PIL import Image

# 设置默认缓存目录
CACHE_DIR = os.getenv(
    "CRAG_IMAGE_CACHE_DIR",  # 优先从环境变量获取
    os.path.join(os.path.expanduser("~"), ".cache/crag/", "image_cache")  # 默认路径
)
os.makedirs(CACHE_DIR, exist_ok=True)  # 创建缓存目录（如果不存在）

class ImageLoader:
    """图像加载工具类，提供缓存管理功能"""

    def __init__(self, url: str):
        """
        初始化ImageLoader实例
        Args:
            url: 图像的URL地址
        """
        self.url = url

    def _get_cache_filename(self):
        """生成基于URL哈希的缓存文件名"""
        # 获取URL扩展名并转为小写
        file_ext = self.url.split(".")[-1].lower()
        # 使用SHA256哈希生成唯一文件名
        return os.path.join(
            CACHE_DIR,
            sha256(self.url.encode()).hexdigest() + "." + file_ext
        )

    def _save_image_to_cache(self, image: Image.Image):
        """
        将图像保存到本地缓存
        Args:
            image: PIL图像对象
        """
        image.save(self._get_cache_filename())

    def _load_image_from_cache(self):
        """从本地缓存加载图像"""
        return Image.open(self._get_cache_filename())

    def _image_cache_exists(self):
        """检查图像缓存是否存在"""
        return os.path.exists(self._get_cache_filename())

    def download_image(self):
        """从URL下载图像（评估环境会失败，因网络受限）"""
        # 设置请求头模拟浏览器访问
        headers = {"User-Agent": "CRAGBot/v0.0.1"}
        # 发起GET请求获取图像数据
        response = requests.get(
            self.url,
            stream=True,
            timeout=10,  # 超时时间
            headers=headers
        )
        # 成功获取响应时
        if response.status_code == 200:
            # 从响应内容创建图像对象
            image = Image.open(BytesIO(response.content))
            return image
        else:
            # 下载失败抛出异常
            raise Exception(
                f"从 {self.url} 下载图像失败，状态码: {response.status_code}"
            )

    def get_image(self):
        """获取图像（优先从缓存加载，未命中则下载并缓存）"""
        if self._image_cache_exists():
            # 缓存存在时直接加载
            return self._load_image_from_cache()
        else:
            # 缓存不存在时下载并保存到缓存
            image = self.download_image()
            self._save_image_to_cache(image)
            return image