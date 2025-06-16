#!/usr/bin/env python3
"""
CRAG-MM 数据集处理模块
该模块提供处理和批量化CRAG-MM数据集的工具，包含多轮对话（图文混合）
主要组件：
1. CRAGTurnBatchIterator - 将数据集处理为对话回合批次
2. ImageLoader - 处理来自不同来源的图像加载
3. 主执行代码用于调试和演示
"""

import random
from typing import List, Dict, Any, Optional, Tuple, Iterator
import tqdm
from PIL import Image
from datasets import Dataset, load_dataset
from loguru import logger
from utils import download_image_url

# 需要跳过的会话ID列表
SESSIONS_TO_SKIP = [
    "04d98259-27af-41b1-a7be-5798fd1b8e95",
    "695b4b5c-7c65-4f7b-8968-50fe10482a16"
]


class ImageLoader:
    """处理图像加载和缓存的工具类"""

    @staticmethod
    def load_image(conversation_data: Dict[str, Any]) -> Image.Image:
        """
        从对话数据中加载图像，必要时从URL下载
        Args:
            conversation_data: 包含图像数据或URL的字典
        Returns:
            PIL图像对象
        说明：
            - 数据集中要么有'image'字段，要么有'image_url'字段
            - 当无法直接获取图像时，仅提供image_url需要下载
        """
        image = conversation_data.get("image")
        image_url = conversation_data.get("image_url")

        # 如果没有图像但存在URL，则下载图像（带本地缓存）
        if image is None and image_url:
            image_local_path = download_image_url(image_url)  # 下载并缓存
            image = Image.open(image_local_path)  # 打开本地图像

        return image


class CRAGTurnBatchIterator:
    """
    将CRAG-MM数据集处理为对话回合批次
    处理包含多轮对话（图文混合）的复杂结构数据集，转换为适合模型训练/评估的格式
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        """
        初始化批处理迭代器
        Args:
            dataset: HuggingFace数据集（包含CRAG对话数据）
            batch_size: 每个批次包含的对话回合数
            shuffle: 是否打乱对话顺序
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))  # 原始索引列表

        # 如果需要打乱，则随机排序
        if self.shuffle:
            random.shuffle(self.indices)

    def _extract_turn_data(
            self,
            conversation: Dict[str, Any],  # 当前对话数据
            turn_idx: int,  # 当前回合索引
            turn: Dict[str, Any],  # 当前回合数据
            session_id: str,  # 会话ID
            image: Image.Image,  # 图像对象
            image_url: str,  # 图像URL
            image_quality: str,  # 图像质量评级
            answer_lookup: Dict[str, str],  # 答案查找表（interaction_id -> 答案）
            total_turn_count: int,  # 总对话回合数
    ) -> Dict[str, Any]:
        """
        提取单个对话回合的数据
        返回包含结构化数据的字典
        """
        # 提取基础信息
        interaction_id = turn["interaction_id"]
        query = turn["query"]
        query_category = turn["query_category"]
        domain = turn["domain"]
        dynamism = turn["dynamism"]

        # 获取标准答案
        answer = answer_lookup.get(interaction_id, False)
        assert answer, f"未找到interaction_id的答案: {interaction_id}"

        # 提取对话历史（如果存在）
        conversation_history = []
        answer_history = []
        if turn_idx > 0:
            conversation_history = conversation["turns"][:turn_idx]
            # 获取历史答案
            answer_history = [answer_lookup.get(a["interaction_id"], False) for a in conversation_history]
            assert all(answer_history), f"未找到对话历史的答案: {interaction_id}. session_id: {session_id}"

        # 调整图像尺寸（如果是第一视角）
        if image_url is None:
            image = image.resize((960, 1280))

        # 返回结构化数据
        return {
            "session_id": session_id,
            "interaction_id": interaction_id,
            "turn_idx": turn_idx,
            "image": image,
            "image_url": image_url,
            "image_quality": image_quality,
            "query": query,
            "answer": answer,
            "query_category": query_category,
            "domain": domain,
            "dynamism": dynamism,
            "conversation_history": conversation_history,
            "answer_history": answer_history,
            "total_turn_count": total_turn_count,
        }

    def _collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        将单个回合数据整理为批次格式
        Args:
            batch: 包含回合数据的字典列表
        Returns:
            批次化后的数据字典
        """
        # 初始化所有字段的列表
        batch_data = {
            "session_ids": [],
            "interaction_ids": [],
            "turn_idxs": [],
            "images": [],
            "image_urls": [],
            "image_qualities": [],
            "queries": [],
            "answers": [],
            "query_categories": [],
            "domains": [],
            "dynamisms": [],
            "conversation_histories": [],
            "answer_histories": [],
            "total_turn_counts": [],
        }

        # 收集每个项目的数据
        for item in batch:
            for key in batch_data:
                batch_data[key].append(item[key])

        return batch_data

    def __iter__(self):
        """
        迭代数据集生成批次
        确保同一对话的N+1回合严格出现在后续批次中
        """
        from collections import deque

        # 记录每个对话的下一个回合索引
        next_turn_idx = [0] * len(self.dataset)
        # 使用双端队列管理对话索引
        queue = deque(self.indices)

        # 缓存相关数据
        self.conversation_cache = {}  # 对话缓存
        self.answer_lookup_cache = {}  # 答案查找缓存
        self.image_cache = {}  # 图像缓存

        batch = []  # 当前批次

        while queue:
            current_convs = []
            # 获取当前批次需要处理的对话
            while queue and len(current_convs) < self.batch_size:
                conv_id = queue.popleft()
                current_convs.append(conv_id)

            # 处理每个对话的一个回合
            for conv_id in current_convs:
                # 跳过特定会话
                if self.dataset[conv_id]["session_id"] in SESSIONS_TO_SKIP:
                    logger.warning("跳过会话 {}", self.dataset[conv_id]["session_id"])
                    continue

                # 1) 延迟加载（如果需要）
                if conv_id not in self.conversation_cache:
                    # 从数据集中加载
                    conv_data = self.dataset[conv_id]
                    self.conversation_cache[conv_id] = conv_data

                    # 构建答案查找表
                    if isinstance(conv_data, dict):
                        answers = []
                        for idx in range(len(conv_data["answers"]["interaction_id"])):
                            answers.append({
                                "interaction_id": conv_data["answers"]["interaction_id"][idx],
                                "ans_full": conv_data["answers"]["ans_full"][idx],
                            })
                        conv_data["answers"] = answers

                    # 创建答案查找表
                    ans_lookup = {a["interaction_id"]: a["ans_full"] for a in conv_data["answers"]}
                    self.answer_lookup_cache[conv_id] = ans_lookup

                    # 加载并缓存图像
                    self.image_cache[conv_id] = ImageLoader.load_image(conv_data)

                # 2) 从缓存获取数据
                conversation = self.conversation_cache[conv_id]
                answer_lookup = self.answer_lookup_cache[conv_id]
                image = self.image_cache[conv_id]

                # 处理不同格式的对话回合数据
                if isinstance(conversation["turns"], dict):
                    turns = []
                    for idx in range(len(conversation["turns"]["interaction_id"])):
                        _sample = {}
                        for k, v in conversation["turns"].items():
                            _sample[k] = v[idx]
                        turns.append(_sample)
                    conversation["turns"] = turns

                # 获取当前回合信息
                turn_idx = next_turn_idx[conv_id]
                total_turn_count = len(conversation["turns"])
                turn = conversation["turns"][turn_idx]
                image_url = conversation.get("image_url", None)
                image_quality = conversation.get("image_quality", None)

                # 提取回合数据
                turn_data = self._extract_turn_data(
                    conversation=conversation,
                    turn_idx=turn_idx,
                    turn=turn,
                    session_id=conversation["session_id"],
                    image=image,
                    image_url=image_url,
                    image_quality=image_quality,
                    answer_lookup=answer_lookup,
                    total_turn_count=total_turn_count,
                )

                batch.append(turn_data)

                # 3) 更新回合指针
                next_turn_idx[conv_id] += 1

                # 如果还有剩余回合，重新加入队列
                if next_turn_idx[conv_id] < total_turn_count:
                    queue.appendleft(conv_id)  # 左侧添加有助于控制缓存大小
                else:
                    # 清理已完成对话的缓存
                    del self.conversation_cache[conv_id]
                    del self.answer_lookup_cache[conv_id]
                    del self.image_cache[conv_id]

            # 生成批次
            yield self._collate_batch(batch)
            batch = []

        # 处理剩余数据
        if batch:
            yield self._collate_batch(batch)


def main():
    """
    主函数（演示和调试用）
    加载CRAG-MM数据集，通过批处理器处理，并打印每个批次验证实现
    """
    # 加载数据集
    print("加载CRAG-MM数据集...")
    # dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public")
    dataset = load_dataset("crag-mm-2025/crag-mm-multi-turn-public")  # 加载多轮对话数据集
    dataset_split = dataset["validation"]  # 使用验证集

    # 创建批处理器
    print(f"为{len(dataset_split)}个对话创建批处理器...")
    batch_size = 16
    crag_turn_batch_iterator = CRAGTurnBatchIterator(
        dataset=dataset_split,
        batch_size=batch_size,
        shuffle=True  # 打乱顺序
    )

    # 处理批次
    print(f"以{batch_size}为批次处理数据集...")
    for batch in tqdm.tqdm(crag_turn_batch_iterator):
        # 打印批次大小
        num_examples = len(batch["session_ids"])
        print(f"批次包含{num_examples}个示例")
        print(f"缓存大小: {len(crag_turn_batch_iterator.conversation_cache)}")
        # 取消注释以打印完整批次详情：
        # print(batch)


if __name__ == "__main__":
    main()