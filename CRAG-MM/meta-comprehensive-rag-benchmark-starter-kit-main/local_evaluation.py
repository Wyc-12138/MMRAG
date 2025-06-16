#!/usr/bin/env python3
"""
CRAG-MM数据集评估器脚本
该脚本用于评估用户提供的代理（UserAgent）在CRAG-MM数据集上的表现。
支持生成响应、语义评估（通过OpenAI API）、多轮对话指标计算，以及结果保存。
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

# 设置分词器并行模式
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, load_dataset  # HuggingFace数据集库
from dotenv import load_dotenv  # 环境变量管理
from openai import OpenAI  # OpenAI API
from pydantic import BaseModel  # 数据验证
from rich.console import Console  # 美化终端输出
from rich.panel import Panel
from agents.base_agent import BaseAgent
from agents.user_config import UserAgent  # 用户自定义代理
from crag_batch_iterator import CRAGTurnBatchIterator  # 数据迭代器
from cragmm_search.search import UnifiedSearchPipeline  # 搜索管道
from utils import display_results, ensure_crag_cache_dir_is_configured  # 工具函数
from tokenizers import Tokenizer  # 文本分词
from transformers import AutoTokenizer

# 加载环境变量和缓存目录配置
load_dotenv()
ensure_crag_cache_dir_is_configured()
console = Console()  # 初始化富文本控制台

# 配置常量
DEFAULT_EVAL_MODEL = "gpt-4o-mini"  # 默认评估模型
MAX_API_RETRIES = 3  # API最大重试次数
DEFAULT_NUM_WORKERS = 8  # 默认并发工作线程数
MIN_BATCH_SIZE = 1  # 最小批量大小
MAX_BATCH_SIZE = 16  # 最大批量大小
MAX_RESPONSE_LENGTH_IN_TOKENS = 75  # 响应最大长度（token数）


class CRAGTurnEvaluationResult(BaseModel):
    """用于存储单轮对话评估结果的结构化模型"""
    accuracy: bool  # 准确性评分


class CRAGEvaluator:
    """
    CRAG-MM数据集评估器类
    功能：
    1. 生成代理响应
    2. 语义评估（可选）
    3. 计算多轮对话指标
    4. 保存评估结果
    """

    def __init__(
            self,
            dataset: Dataset,
            agent: BaseAgent,
            eval_model_name: str | None = None,
            num_conversations: int | None = None,
            show_progress: bool = True,
            num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        """
        初始化评估器
        参数：
            dataset: 要评估的数据集
            agent: 待评估的代理
            eval_model_name: 用于语义评估的模型名称
            num_conversations: 要评估的对话数量
            show_progress: 是否显示进度条
            num_workers: 并发工作线程数
        """
        # 初始化内部状态
        self.dataset = dataset
        self.agent = agent
        self.eval_model_name = eval_model_name
        self.num_conversations = num_conversations
        self.show_progress = show_progress
        self.num_workers = num_workers

        # 内部变量初始化
        self.batch_iterator: CRAGTurnBatchIterator | None = None
        self.conversations_count: int = 0
        self.agent_response_map: dict[str, str] = {}
        self.all_turn_data: list[dict[str, any]] = []
        self.session_ids_evaluated: set[str] = set()

        # 初始化分词器（用于响应截断）
        tokenizer_path = r"C:/Users/25499/Desktop/MM-RAG/local_models/Llama-3.2-1B-Instruct"
        tokenizer_json_path=os.path.join(tokenizer_path,"tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)
        self.tokenizer.enable_truncation(max_length=MAX_RESPONSE_LENGTH_IN_TOKENS)

    @staticmethod
    def get_system_message() -> str:
        """获取评估器的系统提示信息"""
        return (
            "你是问答系统的专家评估员。"
            "任务是根据真实答案判断预测结果是否正确。"
            "规则：\n"
            "1. 预测包含所有关键信息即正确\n"
            "2. 只要语义相同，不同表述也算正确\n"
            "3. 包含错误信息或遗漏关键信息视为错误\n"
            "输出包含'accuracy'字段的JSON对象"
        )

    def attempt_api_call(
            self,
            client: OpenAI,
            model_name: str,
            messages: list,
            max_retries: int = MAX_API_RETRIES,
    ) -> CRAGTurnEvaluationResult | None:
        """
        尝试调用OpenAI API（带重试机制）
        参数：
            client: OpenAI客户端实例
            model_name: 模型名称
            messages: 对话消息列表
            max_retries: 最大重试次数
        返回：
            成功时返回评估结果，失败返回None
        """
        for attempt in range(max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=CRAGTurnEvaluationResult,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                error_message = f"API调用失败（第{attempt + 1}/{max_retries}次）：{str(e)}"
                if attempt == max_retries - 1:
                    console.print(f"[red]经过{MAX_API_RETRIES}次尝试后失败：{str(e)}[/red]")
                else:
                    console.print(f"[yellow]{error_message}，正在重试...[/yellow]")
        return None

    def evaluate_response(self, crag_turn_data: dict[str, any]) -> dict[str, any]:
        """
        评估单个响应
        参数：
            crag_turn_data: 包含查询、真实答案和代理响应的字典
        返回：
            添加评估结果的字典
        """
        agent_response = crag_turn_data["agent_response"]
        ground_truth = crag_turn_data["ground_truth"]
        query = crag_turn_data["query"]

        # 基础判断：是否包含"我不知道"或完全匹配
        is_idk = "i don't know" in agent_response.lower()
        is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
        is_semantically_correct = False
        api_response = None

        # 初始正确性判断基于完全匹配
        is_correct = is_exact_match

        # 如果未禁用语义评估且未命中"我不知道"且非完全匹配
        if not is_idk and not is_exact_match and self.eval_model_name:
            local_openai_client = OpenAI()
            messages = [
                {"role": "system", "content": self.get_system_message()},
                {"role": "user", "content": f"问题: {query}\n真实答案: {ground_truth}\n预测结果: {agent_response}"},
            ]
            api_response = self.attempt_api_call(local_openai_client, self.eval_model_name, messages)
            if api_response:
                is_semantically_correct = api_response.accuracy
                is_correct = is_semantically_correct

        if is_exact_match:
            is_semantically_correct = True  # 完全匹配视为语义正确

        return {
            **crag_turn_data,
            "is_exact_match": is_exact_match,
            "is_correct": is_correct,
            "is_miss": is_idk,
            "is_semantically_correct": is_semantically_correct,
            "api_response": api_response.model_dump() if api_response else None,
        }

    def initialize_evaluation(self) -> None:
        """初始化评估所需变量"""
        console.print(f"[blue]开始评估，使用{self.num_workers}个工作线程[/blue]")
        if self.eval_model_name:
            console.print(f"[blue]使用语义评估模型：{self.eval_model_name}[/blue]")

        # 确定评估的对话数量
        self.conversations_count = len(self.dataset) if self.num_conversations is None else min(self.num_conversations,
                                                                                                len(self.dataset))

        # 确定批量大小
        batch_size = int(np.clip(self.agent.get_batch_size(), MIN_BATCH_SIZE, MAX_BATCH_SIZE))

        # 初始化内部状态
        self.agent_response_map = {}
        self.all_turn_data = []
        self.session_ids_evaluated = set()

        # 实例化批量迭代器
        self.batch_iterator = CRAGTurnBatchIterator(dataset=self.dataset, batch_size=batch_size, shuffle=False)

    def generate_agent_responses(self, progress_callback: Callable[[int, int], None] = None) -> None:
        """
        第一阶段：为数据集中的每个回合生成代理响应
        参数：
            progress_callback: 进度回调函数
        """
        if self.batch_iterator is None:
            raise ValueError("批处理迭代器未初始化，请先调用initialize_evaluation()")

        for batch_idx, batch in enumerate(
                tqdm.tqdm(self.batch_iterator, desc="生成响应", disable=not self.show_progress)):
            # 获取当前批次数据
            interaction_ids = batch["interaction_ids"]
            queries = batch["queries"]
            images = batch["images"]
            conversation_histories = batch["conversation_histories"]

            # 构建消息历史记录（用于多轮对话）
            message_histories = []
            interaction_id_histories = []
            for conversation_history in conversation_histories:
                message_history = []
                interaction_id_history = []
                for turn in conversation_history:
                    turn_interaction_id = turn["interaction_id"]
                    turn_agent_response = self.agent_response_map.get(turn_interaction_id)
                    if not turn_agent_response:
                        raise AssertionError(
                            f"未找到回合{turn_interaction_id}的代理响应。"
                            "是否意外打乱了多轮对话？"
                        )
                    message_history.append({"role": "user", "content": turn["query"]})
                    message_history.append({"role": "assistant", "content": turn_agent_response})
                    interaction_id_history.append(turn_interaction_id)
                message_histories.append(message_history)
                interaction_id_histories.append(interaction_id_history)

            # 生成当前批次的响应
            agent_responses = self.agent.batch_generate_response(queries, images, message_histories)
            agent_responses = self.truncate_agent_responses(agent_responses)  # 截断过长的响应

            # 收集响应并添加评估数据
            for idx, interaction_id in enumerate(interaction_ids):
                agent_response = agent_responses[idx]
                self.agent_response_map[interaction_id] = agent_response
                self.all_turn_data.append({
                    "session_id": batch["session_ids"][idx],
                    "interaction_id": interaction_id,
                    "turn_idx": batch["turn_idxs"][idx],
                    "is_ego": batch["image_urls"][idx] is None,
                    "image_quality": batch["image_qualities"][idx],
                    "query_category": batch["query_categories"][idx],
                    "domain": batch["domains"][idx],
                    "dynamism": batch["dynamisms"][idx],
                    "query": queries[idx],
                    "ground_truth": batch["answers"][idx],
                    "agent_response": agent_response,
                    "total_turn_count": batch["total_turn_counts"][idx],
                    "interaction_id_history": interaction_id_histories[idx]
                })
                self.session_ids_evaluated.add(batch["session_ids"][idx])

            if progress_callback:
                conversations_evaluated = len(self.session_ids_evaluated)
                progress_callback(conversations_evaluated, self.conversations_count)

            if len(self.session_ids_evaluated) > self.conversations_count:
                console.print(f"[yellow]已评估{len(self.session_ids_evaluated)}个对话，提前终止评估[/yellow]")
                break

    def evaluate_agent_responses(
            self,
            turn_data: list[dict[str, any]],
            progress_callback: Callable[[int, int], None] = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
        """
        第二阶段：评估代理响应并计算得分
        参数：
            turn_data: 包含代理响应的回合数据
        返回：
            包含评估结果和得分字典的元组
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.evaluate_response, data) for data in turn_data]
            for future_idx, future in tqdm.tqdm(enumerate(as_completed(futures)), total=len(futures), desc="评估响应",
                                                disable=not self.show_progress):
                results.append(future.result())
                if progress_callback is not None:
                    progress_callback(future_idx, len(turn_data))

        # 将结果转换为DataFrame
        turn_evaluation_results_df = pd.DataFrame(results)
        turn_evaluation_results_df = turn_evaluation_results_df.sort_values(by=["session_id", "turn_idx"])

        # 分别处理所有数据和ego数据
        ego_turn_evaluation_results_df = turn_evaluation_results_df[turn_evaluation_results_df["is_ego"] == True]
        all_scores_dictionary = self.calculate_scores(turn_evaluation_results_df)
        ego_scores_dictionary = self.calculate_scores(ego_turn_evaluation_results_df)

        turn_evaluation_results = {"all": turn_evaluation_results_df, "ego": ego_turn_evaluation_results_df}
        score_dictionaries = {"all": all_scores_dictionary, "ego": ego_scores_dictionary}
        return turn_evaluation_results, score_dictionaries

    def calculate_scores(self, turn_evaluation_results_df: pd.DataFrame) -> dict[str, float]:
        """
        计算得分（包括单轮和多轮对话指标）
        参数：
            turn_evaluation_results_df: 包含评估结果的DataFrame
        返回：
            包含各项指标的字典
        """
        multi_turn_conversation_score_map: dict[str, float] = {}

        def _set_is_correct_false_after_consecutive(group: pd.DataFrame) -> pd.DataFrame:
            """
            在连续错误响应后标记为不正确
            并计算多轮对话得分
            """
            group_copy = group.copy().reset_index(drop=True)
            for i in range(1, len(group_copy)):
                if not group_copy.loc[i - 1, 'is_correct'] and not group_copy.loc[i, 'is_correct']:
                    group_copy.loc[i + 1:, 'is_correct'] = False
                    group_copy.loc[i + 1:, 'is_exact_match'] = False
                    group_copy.loc[i + 1:, 'is_miss'] = True
                    group_copy.loc[i + 1:, 'is_semantically_correct'] = False
                    break
            group_copy["is_hallucination"] = ~group_copy["is_correct"] & ~group_copy["is_miss"]
            multi_turn_conversation_score = group_copy["is_correct"].mean() - group_copy["is_hallucination"].mean()
            group_copy["multi_turn_conversation_score"] = multi_turn_conversation_score
            session_id = group_copy.iloc[0]["session_id"]
            multi_turn_conversation_score_map[session_id] = multi_turn_conversation_score
            return group_copy

        # 按对话分组计算多轮对话得分
        turn_evaluation_results_df = turn_evaluation_results_df.groupby("session_id", group_keys=False)[
            turn_evaluation_results_df.columns].apply(_set_is_correct_false_after_consecutive)

        # 计算各项指标
        total = len(turn_evaluation_results_df)
        correct_exact = turn_evaluation_results_df["is_exact_match"].sum()
        correct = turn_evaluation_results_df["is_correct"].sum()
        miss = turn_evaluation_results_df["is_miss"].sum()
        hallucination = total - (correct + miss)

        exact_match = correct_exact / total
        accuracy = correct / total
        missing = miss / total
        hallucination_rate = hallucination / total
        truthfulness_score = ((2 * correct + miss) / total) - 1 if total > 1 else 0.0
        mean_multi_turn_conversation_score = np.mean(list(multi_turn_conversation_score_map.values()))

        scores_dictionary = {
            "total": float(total),
            "correct_exact": float(correct_exact),
            "correct": float(correct),
            "miss": float(miss),
            "hallucination": float(hallucination),
            "exact_match": float(exact_match),
            "accuracy": float(accuracy),
            "missing": float(missing),
            "hallucination_rate": float(hallucination_rate),
            "truthfulness_score": float(truthfulness_score),
            "mean_multi_turn_conversation_score": float(mean_multi_turn_conversation_score)
        }
        return scores_dictionary

    def save_results(self, turn_evaluation_results: dict[str, any], scores_dictionary: dict[str, any],
                     output_dir: str) -> None:
        """
        保存评估结果到指定目录
        参数：
            turn_evaluation_results: 回合评估结果
            scores_dictionary: 得分字典
            output_dir: 输出目录路径
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)
        turn_evaluation_results["all"].to_csv(os.path.join(output_dir, "turn_evaluation_results_all.csv"), index=False)
        turn_evaluation_results["ego"].to_csv(os.path.join(output_dir, "turn_evaluation_results_ego.csv"), index=False)
        with open(os.path.join(output_dir, "scores_dictionary.json"), "w") as f:
            json.dump(scores_dictionary, f, indent=2)

    def evaluate_agent(self) -> tuple[dict[str, any], dict[str, any]]:
        """
        评估代理在数据集上的表现
        返回：
            包含回合评估结果和得分字典的元组
        """
        # 阶段0：初始化评估状态
        self.initialize_evaluation()

        # 阶段1：生成代理响应
        def _generation_progress_callback(conversations_evaluated: int, total_conversations: int) -> None:
            pass

        self.generate_agent_responses(_generation_progress_callback)

        # 阶段2：评估响应
        def _evaluation_progress_callback(turn_evaluated: int, total_turns: int) -> None:
            pass

        turn_evaluation_results, score_dictionaries = self.evaluate_agent_responses(self.all_turn_data,
                                                                                    _evaluation_progress_callback)

        return turn_evaluation_results, score_dictionaries

    def truncate_agent_responses(self, agent_responses: list[str]) -> list[str]:
        """
        截断代理响应到最大允许长度
        参数：
            agent_responses: 代理响应列表
        返回：
            截断后的响应列表
        """
        encodings = self.tokenizer.encode_batch(agent_responses)
        trimmed_agent_responses = [self.tokenizer.decode(enc.ids) for enc in encodings]
        return trimmed_agent_responses


def main() -> None:
    """主函数：解析参数、加载数据集、运行评估并展示结果"""
    parser = argparse.ArgumentParser(description="评估代理在CRAG-MM数据集上的表现")
    parser.add_argument("--dataset-type", type=str, default="single-turn", choices=["single-turn", "multi-turn"],
                        help="数据集类型")
    parser.add_argument("--split", type=str, default="validation", help="数据集分割（'validation', 'public_test'）")
    parser.add_argument("--num-conversations", type=int, default=-1, help="要评估的对话数量（-1表示全部）")
    parser.add_argument("--suppress-web-search-api", action="store_true", help="禁用网络搜索API（用于单源增强任务）")
    parser.add_argument("--display-conversations", type=int, default=10, help="要展示的评估示例数量")
    parser.add_argument("--eval-model", type=str, default=DEFAULT_EVAL_MODEL, help="用于语义评估的OpenAI模型")
    parser.add_argument("--output-dir", type=str, default=None, help="保存结果的路径")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    parser.add_argument("--revision", type=str, default="v0.1.1", help="加载数据集时使用的版本号")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="并行评估的工作线程数")

    args = parser.parse_args()

    # 加载数据集
    console.print(f"[bold blue]加载{args.dataset_type}数据集...[/bold blue]")
    repo_name = f"crag-mm-2025/crag-mm-{args.dataset_type}-public"
    console.print(f"[bold green]从HuggingFace加载：{repo_name}（版本：{args.revision}）[/bold green]")
    dataset = load_dataset(repo_name, revision=args.revision)

    # 确定数据集分割
    available_splits = list(dataset.keys())
    split_to_use = args.split if args.split in available_splits else available_splits[0]
    console.print(f"[bold green]使用分割：'{split_to_use}'，共{len(dataset[split_to_use])}个样本[/bold green]")

    # 处理评估模型参数
    if args.eval_model.lower() == "none":
        args.eval_model = None
        console.print(
            Panel("[bold red]警告：语义评估已禁用[/bold red]\n不会调用LLM作为评估器！\n结果仅依赖精确字符串匹配。",
                  title="[bold red]注意[/bold red]", border_style="red", width=100, padding=(2, 5), expand=False))

    # 处理对话数量参数
    if args.num_conversations == -1:
        args.num_conversations = len(dataset[split_to_use])

    # 配置搜索管道
    search_api_text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    search_api_image_model_name = "openai/clip-vit-large-patch14-336"
    search_api_web_hf_dataset_id = "crag-mm-2025/web-search-index-validation"
    search_api_image_hf_dataset_id = "crag-mm-2025/image-search-index-validation"

    # 如果禁用网络搜索API
    if args.suppress_web_search_api:
        search_api_web_hf_dataset_id = None

    # 创建搜索管道实例
    search_pipeline = UnifiedSearchPipeline(
        text_model_name=search_api_text_model_name,
        image_model_name=search_api_image_model_name,
        web_hf_dataset_id=search_api_web_hf_dataset_id,
        image_hf_dataset_id=search_api_image_hf_dataset_id,
    )

    # 创建评估器实例
    evaluator = CRAGEvaluator(
        dataset=dataset[split_to_use],
        agent=UserAgent(search_pipeline=search_pipeline),
        eval_model_name=args.eval_model,
        num_conversations=args.num_conversations,
        show_progress=not args.no_progress,
        num_workers=args.num_workers,
    )

    # 运行评估
    turn_evaluation_results, score_dictionaries = evaluator.evaluate_agent()

    # 显示结果
    display_results(
        console,
        turn_evaluation_results["all"],
        score_dictionaries["all"],
        display_conversations=args.display_conversations,
        is_ego=False,
        is_multi_turn=(args.dataset_type == "multi-turn"),
    )

    # 显示ego数据结果
    if len(turn_evaluation_results["ego"]) > 0:
        display_results(
            console,
            turn_evaluation_results["ego"],
            score_dictionaries["ego"],
            display_conversations=args.display_conversations,
            is_ego=True,
            is_multi_turn=(args.dataset_type == "multi-turn"),
        )

    # 保存结果
    if args.output_dir:
        evaluator.save_results(turn_evaluation_results, score_dictionaries, args.output_dir)


if __name__ == "__main__":
    main()