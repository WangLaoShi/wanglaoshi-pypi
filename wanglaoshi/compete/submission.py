"""Compete: 提交文件生成（从 CompetitionTools 拆出，行为不变）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    提交文件生成器
    
    用于数据科学比赛（如Kaggle）中生成符合格式要求的提交文件。
    支持单模型提交和多模型融合提交。
    
    使用示例:
        generator = SubmissionGenerator()
        # 生成单模型提交文件
        generator.create_submission(y_pred, 'sample_submission.csv', 'my_submission.csv')
        # 多模型融合
        generator.ensemble_submissions(['sub1.csv', 'sub2.csv', 'sub3.csv'], 
                                     weights=[0.4, 0.3, 0.3], 
                                     output_path='ensemble_submission.csv')
    """
    
    def __init__(self):
        """初始化提交文件生成器"""
        pass
    
    @staticmethod
    def _resolve_id_and_target_columns(
        sample_submission: pd.DataFrame,
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> tuple:
        """
        根据样本提交文件解析 ID 列与目标列。

        优先级：
        1. 调用方显式传入且存在于模板中的列名
        2. 常见 ID 列名候选
        3. 模板第一列作为 ID，其余列作为目标列
        """
        cols = list(sample_submission.columns)
        if len(cols) < 2:
            raise ValueError(f"样本提交文件至少需要 2 列（ID + 目标），当前列: {cols}")

        # --- ID 列 ---
        resolved_id = None
        if id_column is not None:
            if id_column not in sample_submission.columns:
                raise ValueError(
                    f"指定的 id_column='{id_column}' 不在样本列中: {cols}"
                )
            resolved_id = id_column
        else:
            for cand in ['id', 'Id', 'ID', 'index', 'Index']:
                if cand in sample_submission.columns:
                    resolved_id = cand
                    break
            if resolved_id is None:
                resolved_id = cols[0]

        # --- 目标列 ---
        remaining = [c for c in cols if c != resolved_id]
        if not remaining:
            raise ValueError("样本提交文件缺少目标列")

        if target_column is not None:
            if target_column not in sample_submission.columns:
                raise ValueError(
                    f"指定的 target_column='{target_column}' 不在样本列中: {cols}"
                )
            if target_column == resolved_id:
                raise ValueError("target_column 不能与 id_column 相同")
            resolved_targets = [target_column]
        else:
            # 默认：除 ID 外的全部列（兼容单目标与多输出模板）
            resolved_targets = remaining

        return resolved_id, resolved_targets

    @staticmethod
    def _to_prediction_array(predictions: Union[np.ndarray, pd.Series, List, pd.DataFrame]) -> np.ndarray:
        """将预测结果统一转为 numpy 数组。"""
        if isinstance(predictions, pd.DataFrame):
            return predictions.to_numpy()
        if isinstance(predictions, pd.Series):
            return predictions.to_numpy()
        if isinstance(predictions, list):
            return np.asarray(predictions)
        if isinstance(predictions, np.ndarray):
            return predictions
        raise TypeError(f"不支持的预测结果类型: {type(predictions)}")

    def create_submission(
        self,
        predictions: Union[np.ndarray, pd.Series, List, pd.DataFrame],
        sample_submission_path: str,
        output_path: str = 'submission.csv',
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
        prediction_type: str = 'auto',
        class_labels: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """
        生成提交文件（按样本模板保持列名与顺序）。

        参数:
            predictions: 预测结果（1D 标签/数值，或 2D 概率/多输出）
            sample_submission_path: 样本提交文件路径
            output_path: 输出路径
            id_column: ID 列名；None 时自动推断
            target_column: 单目标列名；None 时使用模板中除 ID 外的全部目标列
            prediction_type:
                - 'auto': 按形状自动判断（1D=label/value，2D且列数匹配模板=概率/多输出）
                - 'label': 一维类别/数值标签
                - 'proba': 二维概率；若模板只有单目标列，需配合 class_labels 映射或显式转换
                - 'multioutput': 二维多输出，列数必须与模板目标列数一致
            class_labels: 当 prediction_type='proba' 且模板只要单列标签时，
                          用 argmax 后映射到这些标签；若不提供则写出整数类别索引，
                          **不会静默假设模板需要概率列**

        注意:
            不会再因为「模板有 target 列」就静默对 2D 概率做 argmax。
            需要标签时请使用 prediction_type='label'，或提供 class_labels。
        """
        try:
            sample_submission = pd.read_csv(sample_submission_path)
            logger.info(f"读取样本提交文件: {sample_submission_path}")
            logger.info(f"样本文件列名: {sample_submission.columns.tolist()}")

            resolved_id, resolved_targets = self._resolve_id_and_target_columns(
                sample_submission, id_column=id_column, target_column=target_column
            )
            submission_ids = sample_submission[resolved_id].values
            n = len(submission_ids)

            arr = self._to_prediction_array(predictions)
            if arr.size == 0:
                raise ValueError("预测结果为空")

            # 统一为 2D 便于处理
            if arr.ndim == 1:
                arr_2d = arr.reshape(-1, 1)
                inferred = 'label'
            elif arr.ndim == 2:
                arr_2d = arr
                inferred = 'multioutput' if arr.shape[1] == len(resolved_targets) else 'proba'
            else:
                raise ValueError(f"不支持的预测维度: {arr.ndim}，仅支持 1D/2D")

            if arr_2d.shape[0] != n:
                raise ValueError(
                    f"预测结果行数 ({arr_2d.shape[0]}) 与 ID 数量 ({n}) 不匹配"
                )

            mode = prediction_type
            if mode == 'auto':
                mode = inferred

            submission_df = pd.DataFrame({resolved_id: submission_ids})

            if mode == 'label':
                if arr_2d.shape[1] != 1:
                    raise ValueError(
                        "prediction_type='label' 需要 1 列预测；"
                        "若输入为概率矩阵，请先转换或使用 prediction_type='proba' 并提供 class_labels"
                    )
                if len(resolved_targets) != 1:
                    raise ValueError(
                        f"模板有 {len(resolved_targets)} 个目标列，无法用单列 label 填充: {resolved_targets}"
                    )
                submission_df[resolved_targets[0]] = arr_2d[:, 0]

            elif mode == 'proba':
                # 概率矩阵：仅当模板目标列数等于类别数时按列写出；
                # 若模板只有 1 个目标列，必须显式用 class_labels（或用户先 argmax）
                if arr_2d.shape[1] < 2:
                    raise ValueError("prediction_type='proba' 需要至少 2 列概率")
                if len(resolved_targets) == 1:
                    if class_labels is None:
                        raise ValueError(
                            "模板只有单目标列，但收到概率矩阵。"
                            "请先自行转换为标签，或传入 class_labels 以安全映射 argmax 结果；"
                            "禁止静默 argmax。"
                        )
                    if len(class_labels) != arr_2d.shape[1]:
                        raise ValueError(
                            f"class_labels 长度 ({len(class_labels)}) 与概率列数 ({arr_2d.shape[1]}) 不一致"
                        )
                    idx = np.argmax(arr_2d, axis=1)
                    submission_df[resolved_targets[0]] = [class_labels[i] for i in idx]
                elif len(resolved_targets) == arr_2d.shape[1]:
                    for i, col in enumerate(resolved_targets):
                        submission_df[col] = arr_2d[:, i]
                else:
                    raise ValueError(
                        f"概率列数 ({arr_2d.shape[1]}) 与模板目标列数 ({len(resolved_targets)}) 不一致: "
                        f"{resolved_targets}"
                    )

            elif mode == 'multioutput':
                if arr_2d.shape[1] != len(resolved_targets):
                    raise ValueError(
                        f"multioutput 列数 ({arr_2d.shape[1]}) 与模板目标列 "
                        f"({len(resolved_targets)}) 不一致"
                    )
                for i, col in enumerate(resolved_targets):
                    submission_df[col] = arr_2d[:, i]
            else:
                raise ValueError(
                    f"不支持的 prediction_type: {prediction_type}，"
                    f"可选: auto/label/proba/multioutput"
                )

            # 按样本模板列顺序重排
            submission_df = submission_df[sample_submission.columns.tolist()]
            submission_df.to_csv(output_path, index=False)
            logger.info(f"提交文件已生成: {output_path}，形状={submission_df.shape}")
            return submission_df

        except FileNotFoundError:
            logger.error(f"未找到样本提交文件: {sample_submission_path}")
            raise
        except Exception as e:
            logger.error(f"生成提交文件时出错: {str(e)}")
            raise
    
    def ensemble_submissions(self,
                            submission_files: List[str],
                            weights: Optional[List[float]] = None,
                            method: str = 'weighted_average',
                            output_path: str = 'ensemble_submission.csv',
                            sample_submission_path: Optional[str] = None) -> pd.DataFrame:
        """
        融合多个提交文件
        
        支持多种融合方法：
        - weighted_average: 加权平均（默认）
        - simple_average: 简单平均
        - rank_average: 排名平均（适合分类问题）
        - median: 中位数融合
        
        参数:
            submission_files: 要融合的提交文件路径列表
            weights: 权重列表（用于加权平均），如果为None则使用简单平均
            method: 融合方法
                   - 'weighted_average': 加权平均
                   - 'simple_average': 简单平均
                   - 'rank_average': 排名平均
                   - 'median': 中位数
            output_path: 输出文件路径
            sample_submission_path: 样本提交文件路径（用于获取格式），如果为None则使用第一个文件
            
        返回:
            融合后的提交文件DataFrame
            
        示例:
            # 融合3个提交文件，使用加权平均
            ensemble = generator.ensemble_submissions(
                ['sub1.csv', 'sub2.csv', 'sub3.csv'],
                weights=[0.4, 0.3, 0.3],
                output_path='ensemble_submission.csv'
            )
        """
        try:
            if len(submission_files) < 2:
                raise ValueError("至少需要2个提交文件才能进行融合")
            
            # 读取所有提交文件
            submissions = []
            for file_path in submission_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"未找到提交文件: {file_path}")
                df = pd.read_csv(file_path)
                submissions.append(df)
                logger.info(f"读取提交文件: {file_path}, 形状: {df.shape}")
            
            # 检查所有文件是否有相同的列
            first_cols = set(submissions[0].columns)
            for i, df in enumerate(submissions[1:], 1):
                if set(df.columns) != first_cols:
                    raise ValueError(f"提交文件 {submission_files[i]} 的列与第一个文件不匹配")
            
            # 确定ID列和目标列
            id_column = None
            target_column = None
            
            # 尝试识别ID列（通常是第一列或名为'id'/'Id'/'ID'的列）
            possible_id_cols = ['id', 'Id', 'ID', 'index', 'Index']
            for col in possible_id_cols:
                if col in submissions[0].columns:
                    id_column = col
                    break
            
            if id_column is None:
                # 使用第一列作为ID列
                id_column = submissions[0].columns[0]
            
            # 目标列是除了ID列之外的其他列
            target_columns = [col for col in submissions[0].columns if col != id_column]
            
            if len(target_columns) == 0:
                raise ValueError("未找到目标列")
            
            # 获取ID
            ids = submissions[0][id_column].values
            
            # 检查所有文件的ID是否一致
            for i, df in enumerate(submissions[1:], 1):
                if not np.array_equal(df[id_column].values, ids):
                    raise ValueError(f"提交文件 {submission_files[i]} 的ID与第一个文件不匹配")
            
            # 融合预测结果
            ensemble_df = pd.DataFrame({id_column: ids})
            
            for target_col in target_columns:
                # 收集所有文件的该列预测值
                predictions = np.array([df[target_col].values for df in submissions])
                
                if method == 'weighted_average':
                    # 加权平均
                    if weights is None:
                        weights = [1.0 / len(submissions)] * len(submissions)
                    else:
                        if len(weights) != len(submissions):
                            raise ValueError(f"权重数量 ({len(weights)}) 与提交文件数量 ({len(submissions)}) 不匹配")
                        # 归一化权重
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                    
                    ensemble_pred = np.average(predictions, axis=0, weights=weights)
                    logger.info(f"使用加权平均融合 {target_col}，权重: {weights}")
                    
                elif method == 'simple_average':
                    # 简单平均
                    ensemble_pred = np.mean(predictions, axis=0)
                    logger.info(f"使用简单平均融合 {target_col}")
                    
                elif method == 'rank_average':
                    # 排名平均（适合分类问题）
                    # 将预测值转换为排名，然后平均排名，再转换回预测值
                    ranks = np.array([np.argsort(np.argsort(pred)) for pred in predictions])
                    avg_ranks = np.mean(ranks, axis=0)
                    # 将平均排名转换回预测值范围
                    min_val = np.min([np.min(pred) for pred in predictions])
                    max_val = np.max([np.max(pred) for pred in predictions])
                    denom = max(len(avg_ranks) - 1, 1)
                    ensemble_pred = min_val + (avg_ranks / denom) * (max_val - min_val)
                    logger.info(f"使用排名平均融合 {target_col}")
                    
                elif method == 'median':
                    # 中位数融合
                    ensemble_pred = np.median(predictions, axis=0)
                    logger.info(f"使用中位数融合 {target_col}")
                    
                else:
                    raise ValueError(f"不支持的融合方法: {method}")
                
                ensemble_df[target_col] = ensemble_pred
            
            # 保存融合后的提交文件
            ensemble_df.to_csv(output_path, index=False)
            logger.info(f"融合提交文件已生成: {output_path}")
            logger.info(f"融合了 {len(submissions)} 个提交文件")
            
            return ensemble_df
            
        except Exception as e:
            logger.error(f"融合提交文件时出错: {str(e)}")
            raise
    
    def validate_submission(
        self,
        submission_path: str,
        sample_submission_path: str,
        check_values: bool = True,
        id_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        验证提交文件格式是否正确（与 create_submission 共用列解析规则）。

        数值列使用数值统计；字符串/类别标签使用类别统计，避免对标签做 min/mean 误判。
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        try:
            submission = pd.read_csv(submission_path)
            sample = pd.read_csv(sample_submission_path)

            result['info']['submission_shape'] = submission.shape
            result['info']['sample_shape'] = sample.shape

            resolved_id, resolved_targets = self._resolve_id_and_target_columns(
                sample, id_column=id_column, target_column=target_column
            )
            expected_cols = sample.columns.tolist()

            if list(submission.columns) != expected_cols:
                # 允许列集合相同但顺序不同时给出明确错误（要求顺序一致）
                if set(submission.columns) != set(expected_cols):
                    result['is_valid'] = False
                    result['errors'].append(
                        f"列名不匹配: 提交={list(submission.columns)}, 样本={expected_cols}"
                    )
                else:
                    result['is_valid'] = False
                    result['errors'].append(
                        f"列顺序不匹配: 提交={list(submission.columns)}, 样本={expected_cols}"
                    )

            if len(submission) != len(sample):
                result['is_valid'] = False
                result['errors'].append(
                    f"行数不匹配: 提交={len(submission)}, 样本={len(sample)}"
                )

            if resolved_id not in submission.columns:
                result['is_valid'] = False
                result['errors'].append(f"未找到 ID 列: {resolved_id}")
            elif len(submission) == len(sample):
                if not submission[resolved_id].astype(str).equals(sample[resolved_id].astype(str)):
                    result['is_valid'] = False
                    result['errors'].append("ID 列内容不匹配")

            if check_values and result['is_valid']:
                for col in resolved_targets:
                    if col not in submission.columns:
                        result['is_valid'] = False
                        result['errors'].append(f"缺少目标列: {col}")
                        continue

                    series = submission[col]
                    nan_count = int(series.isna().sum())
                    if nan_count > 0:
                        result['is_valid'] = False
                        result['errors'].append(f"列 {col} 包含 {nan_count} 个缺失值")

                    # 数值列：数值统计；非数值：类别统计
                    if pd.api.types.is_numeric_dtype(series):
                        values = series.to_numpy(dtype=float)
                        if np.isinf(values).any():
                            result['is_valid'] = False
                            result['errors'].append(f"列 {col} 包含无穷大值")
                        finite = values[np.isfinite(values)]
                        result['info'][f'{col}_stats'] = {
                            'dtype': 'numeric',
                            'min': float(np.min(finite)) if len(finite) else None,
                            'max': float(np.max(finite)) if len(finite) else None,
                            'mean': float(np.mean(finite)) if len(finite) else None,
                            'std': float(np.std(finite)) if len(finite) else None,
                        }
                        if len(finite) and (finite.min() < -1e10 or finite.max() > 1e10):
                            result['warnings'].append(
                                f"列 {col} 数值范围异常: [{finite.min()}, {finite.max()}]"
                            )
                    else:
                        vc = series.astype(str).value_counts(dropna=False)
                        result['info'][f'{col}_stats'] = {
                            'dtype': 'categorical',
                            'nunique': int(series.nunique(dropna=False)),
                            'top_values': vc.head(10).to_dict(),
                        }

            logger.info(f"提交文件验证完成: {'通过' if result['is_valid'] else '失败'}")
            return result

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"验证过程出错: {str(e)}")
            logger.error(f"验证提交文件时出错: {str(e)}")
            return result
