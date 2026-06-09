import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        运行主LLM生成循环 - 实现多轮搜索+推理的核心逻辑

        功能说明：
        1. 支持多轮对话（最多max_turns轮）
        2. 每轮可以调用搜索工具获取外部知识
        3. 跟踪每个样本的活跃状态、搜索次数、轮次统计
        4. 最终强制所有样本给出答案（不允许搜索）

        Args:
            gen_batch: 输入批次数据，包含input_ids、attention_mask等
            initial_input_ids: 初始输入的token ids [batch_size, seq_len]

        Returns:
            Tuple[Dict, Dict]: 返回最终输出和元信息
        """

        # ========== 初始化左右两侧状态 ==========
        # 左侧：固定不变的原始prompt（用于最终输出时保留问题）
        # 截取最后max_start_length个token作为左侧固定部分
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}

        # 右侧：动态增长的响应部分（包含模型生成的所有内容）
        # 初始化为空tensor，逐步追加模型的响应和检索结果
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        # ========== 初始化追踪状态 ==========
        # active_mask: 标记哪些样本还在活跃状态（未完成）
        # True表示该样本仍在进行中，False表示已完成（已给出答案）
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)

        # turns_stats: 统计每个样本经历的轮次（初始为1，表示第一轮）
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)

        # valid_action_stats: 统计每个样本的有效动作次数（搜索或答案）
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)

        # valid_search_stats: 统计每个样本的有效搜索次数
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)

        # active_num_list: 记录每轮结束后仍活跃的样本数量（用于监控训练进度）
        active_num_list = [active_mask.sum().item()]

        # rollings: 当前的滚动状态，包含所有样本的对话历史
        rollings = gen_batch

        # ========== 主生成循环：支持多轮搜索 ==========
        # 每轮：模型生成 → 判断动作（搜索/答案）→ 执行动作 → 更新状态
        for step in range(self.config.max_turns):
            # 如果所有样本都已完成（给出答案），提前结束循环
            if not active_mask.sum():
                break

            # 优化：将批次数据裁剪到有效长度，去除多余的padding
            # 这样可以减少计算量，提高生成效率
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # ========== 提取活跃样本进行生成 ==========
            # 只对仍活跃的样本进行生成，已完成的样本不参与计算
            # 使用active_mask过滤rollings中的活跃样本
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            # 调用生成模型，使用GPU padding优化（处理多GPU场景）
            # 返回的gen_output包含模型生成的responses
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # 提取元信息（包含生成过程的统计数据）
            meta_info = gen_output.meta_info

            # ========== 后处理模型响应 ==========
            # _postprocess_responses: 确保响应在</search>或</answer>处截断
            # 返回token ids和字符串两种格式
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            # 将活跃样本的响应对齐到完整批次维度
            # 非活跃样本的位置用padding填充
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # ========== 执行预测动作（环境交互） ==========
            # execute_predictions: 解析模型输出，执行搜索或答案动作
            # 返回：
            # - next_obs: 环境返回的观察（检索结果或空字符串）
            # - dones: 是否完成（给出答案后为True）
            # - valid_action: 动作是否有效（search或answer为有效，无效动作返回惩罚）
            # - is_search: 是否为搜索动作
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str,           # 模型生成的响应字符串
                self.tokenizer.pad_token, # padding token
                active_mask              # 活跃样本掩码
            )

            # ========== 更新活跃状态 ==========
            # curr_active_mask: 当前这轮完成后仍活跃的样本
            # 如果样本给出了答案（done=True），则不再活跃
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)

            # 更新全局active_mask：只有既在之前活跃，又在当前活跃的样本才保持活跃
            # 使用逐元素与运算（相当于集合交集）
            active_mask = active_mask * curr_active_mask

            # 记录当前活跃样本数量，用于监控训练进度
            active_num_list.append(active_mask.sum().item())

            # ========== 更新统计信息 ==========
            # turns_stats: 对当前活跃样本的轮次+1
            turns_stats[curr_active_mask] += 1

            # valid_action_stats: 累加有效动作次数
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

            # valid_search_stats: 累加搜索次数
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # ========== 处理观察结果 ==========
            # 将字符串格式的观察（检索结果）转换为token ids
            # 如果超过max_obs_length则截断
            next_obs_ids = self._process_next_obs(next_obs)

            # ========== 更新滚动状态 ==========
            # rollings: 用于下一轮生成的完整输入（prompt + 响应 + 观察）
            # 包含所有历史对话，用于模型理解上下文
            rollings = self._update_rolling_state(
                rollings,        # 当前滚动状态
                responses_ids,   # 模型生成的响应token ids
                next_obs_ids     # 环境返回的观察token ids（检索结果）
            )

            # original_right_side: 用于最终输出的右侧响应
            # 分别保存responses和带info_mask的版本（用于信息遮蔽训练）
            original_right_side = self._update_right_side(
                original_right_side,  # 当前右侧状态
                responses_ids,        # 新增的响应
                next_obs_ids          # 新增的观察
            )

        # ========== 最终LLM生成（强制给出答案） ==========
        # 对于经过max_turns轮仍没有给出答案的样本，强制生成答案
        # do_search=False：不允许搜索，必须给出最终答案
        if active_mask.sum():
            # 优化：裁剪到有效长度
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # 提取仍活跃的样本（未完成答案的样本）
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            # 生成最终答案（不允许搜索）
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # 提取元信息
            meta_info = gen_output.meta_info

            # 后处理响应（确保在</answer>处截断）
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # ========== 执行最终预测（不允许搜索） ==========
            # do_search=False：强制给出答案，不执行搜索操作
            # 返回dones（所有样本都完成）、valid_action、is_search（都为0）
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str,            # 最终答案
                self.tokenizer.pad_token,
                active_mask,
                do_search=False          # 关键：不允许搜索
            )

            # 更新活跃状态（所有样本都应该完成）
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask

            # 记录最终活跃样本数量
            active_num_list.append(active_mask.sum().item())

            # 更新统计信息
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # 更新右侧状态（添加最终答案）
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,       # 最终答案的token ids
                # 注意：这里没有next_obs_ids，因为最终答案不涉及检索
            )

        # ========== 收集元信息 ==========
        # 将统计信息转换为列表格式，便于记录和监控
        meta_info['turns_stats'] = turns_stats.tolist()           # 每个样本的轮次
        meta_info['active_mask'] = active_mask.tolist()           # 最终活跃状态
        meta_info['valid_action_stats'] = valid_action_stats.tolist()  # 有效动作统计
        meta_info['valid_search_stats'] = valid_search_stats.tolist()  # 搜索次数统计

        # 打印活跃轨迹数量（用于调试和监控训练进度）
        # 输出格式：[初始数, 第1轮后, 第2轮后, ..., 最终数]
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # ========== 组装最终输出 ==========
        # 将左右两侧组合成完整的输出格式
        # 左侧：原始问题（固定）
        # 右侧：所有响应和检索结果（动态增长）
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
