"""
Data loading utilities for metaClassifier.

This module handles loading of microbiome data from various sources.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from ..utils.logger import get_logger


class DataLoader:
    """Data loader for microbiome datasets."""
    
    def __init__(self):
        self.logger = get_logger("DataLoader")
        
    def load_data(
        self,
        prof_file: Union[str, Path],
        metadata_file: Union[str, Path],
        scope: Optional[str] = None,
        use_presence_absence: bool = True,
        use_clr: bool = False,
        enable_cohort_analysis: bool = False,
        cohort_column: str = "Project",
        phenotype_col: Optional[str] = None,
        case_phenotype: Optional[str] = None,
        control_phenotype: Optional[str] = None,
        sample_id_col: str = "SampleID",
        project_col: str = "Project",
        group_col: str = "Group",
        label_0: Optional[str] = None,
        label_1: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        统一的load_data接口，兼容build.py的调用方式。
        
        Args:
            prof_file: Profile数据文件路径
            metadata_file: 元数据文件路径
            scope: 范围过滤器，格式为'列名=值'
            use_presence_absence: 是否使用有无数据
            use_clr: 是否应用CLR变换
            enable_cohort_analysis: 是否启用队列分析
            cohort_column: 队列信息列名
            phenotype_col: 表型列名（用于case/control映射）
            case_phenotype: 病例表型值
            control_phenotype: 对照表型值
            sample_id_col: 样本ID列名
            project_col: 项目列名
            group_col: 分组列名
            
        Returns:
            Tuple of (X, y, groups, original_features, constant_removed_features)
        """
        return self.load_microbiome_data(
            profile_path=prof_file,
            metadata_path=metadata_file,
            project=scope,
            use_presence_absence=use_presence_absence,
            use_clr=use_clr,
            enable_cohort_analysis=enable_cohort_analysis,
            cohort_column=cohort_column,
            phenotype_col=phenotype_col,
            case_phenotype=case_phenotype,
            control_phenotype=control_phenotype,
            sample_id_col=sample_id_col,
            project_col=project_col,
            group_col=group_col,
            label_0=label_0,
            label_1=label_1
        )
    
    def load_microbiome_data(
        self,
        profile_path: Union[str, Path],
        metadata_path: Union[str, Path],
        project: Optional[str] = None,
        use_presence_absence: bool = True,
        use_clr: bool = False,
        enable_cohort_analysis: bool = False,
        cohort_column: str = "Project",
        phenotype_col: Optional[str] = None,
        case_phenotype: Optional[str] = None,
        control_phenotype: Optional[str] = None,
        sample_id_col: str = "SampleID",
        project_col: str = "Project",
        group_col: str = "Group",
        label_0: Optional[str] = None,
        label_1: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load microbiome data from CSV files.
        
        Args:
            profile_path: Path to profile data CSV
            metadata_path: Path to metadata CSV
            project: Project name to filter by (optional)
            sample_id_col: Name of sample ID column
            project_col: Name of project column
            group_col: Name of group column
            
        Returns:
            Tuple of (profile_data, labels, metadata)
        """
        self.logger.info(f"Loading data from {profile_path} and {metadata_path}")
        
        # Load profile data (使用read_table以保持与原始项目一致)
        profile_data = pd.read_table(profile_path, sep=',', index_col=0)
        self.logger.info(f"Loaded profile data: {profile_data.shape}")
        
        # Load metadata
        metadata = pd.read_table(metadata_path, sep=',', index_col=0)
        self.logger.info(f"Loaded metadata: {metadata.shape}")
        
        # Filter by project if specified
        if project is not None and project.strip() != '' and project.strip().lower() != 'all':
            # Parse scope string if it contains '='
            if '=' in project:
                column, value = project.split('=', 1)
                metadata = metadata[metadata[column] == value]
                self.logger.info(f"Filtered to {column}='{value}': {metadata.shape}")
            else:
                metadata = metadata[metadata[project_col] == project]
                self.logger.info(f"Filtered to project '{project}': {metadata.shape}")
        else:
            self.logger.info("No project filter applied, using all data")
        
        # Get common samples (确保顺序一致性)
        common_samples = sorted(set(profile_data.index) & set(metadata.index))
        self.logger.info(f"Found {len(common_samples)} common samples")
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between profile and metadata")
        
        # Filter data to common samples (保持排序顺序)
        profile_data = profile_data.loc[common_samples]
        metadata = metadata.loc[common_samples]
        
        # Generate labels (支持phenotype映射和自定义标签映射)
        labels = self._generate_labels(metadata, group_col, phenotype_col, case_phenotype, control_phenotype, label_0, label_1)
        
        # 记录原始特征名称（在数据处理之前）
        original_features = profile_data.columns.tolist()
        
        # 对于单个项目，先移除在该项目样本中为常量的特征
        if project is not None:
            self.logger.info(f"针对项目 '{project}' 进行特征筛选...")
            profile_data, project_constant_removed = self._remove_constant_features_with_tracking(profile_data)
            self.logger.info(f"项目特定常量特征移除: {len(project_constant_removed)} 个特征")
        else:
            project_constant_removed = []
        
        # 应用数据处理
        processed_data, constant_removed_features = self._process_data_with_tracking(profile_data, use_presence_absence, use_clr)
        
        # 合并所有被移除的特征
        all_constant_removed = project_constant_removed + constant_removed_features
        
        # Generate groups (for cohort analysis)
        if enable_cohort_analysis:
            groups = self._generate_groups(metadata, cohort_column)
        else:
            groups = None
        
        self.logger.info(f"Final dataset: {processed_data.shape}, labels: {labels.shape}")
        self.logger.info(f"Label distribution: {np.unique(labels, return_counts=True)}")
        if groups is not None:
            self.logger.info(f"Group distribution: {np.unique(groups, return_counts=True)}")
        
        # 保存样本ID信息供后续使用
        self.last_sample_ids_ = processed_data.index.tolist()
        
        # 打印数据信息
        self._print_data_info(processed_data, groups)
        
        return processed_data, labels, groups, original_features, all_constant_removed
    
    def _generate_labels(self, metadata: pd.DataFrame, group_col: str, 
                        phenotype_col: Optional[str] = None, 
                        case_phenotype: Optional[str] = None, 
                        control_phenotype: Optional[str] = None,
                        label_0: Optional[str] = None,
                        label_1: Optional[str] = None) -> np.ndarray:
        """Generate binary labels from group column or phenotype mapping."""
        # 优先使用 phenotype 映射，否则回退到已有 Group 列
        if phenotype_col and phenotype_col in metadata.columns and case_phenotype is not None and control_phenotype is not None:
            # 规范大小写与空白
            col = metadata[phenotype_col].astype(str).str.strip().str.lower()
            case_norm = str(case_phenotype).strip().lower()
            ctrl_norm = str(control_phenotype).strip().lower()
            labels = np.where(col == case_norm, 1, np.where(col == ctrl_norm, 0, np.nan))
            
            # 严格二分类：仅保留标注为 case/control 的样本
            valid_mask = ~np.isnan(labels)
            if not valid_mask.any():
                raise ValueError(f"No valid samples found with phenotype mapping: case='{case_phenotype}', control='{control_phenotype}'")
            
            # 过滤数据
            labels = labels[valid_mask].astype(int)
            self.logger.info(f"Phenotype mapping using '{phenotype_col}': case='{case_phenotype}', control='{control_phenotype}'")
            self.logger.info(f"Kept samples: {len(labels)}; Dropped (non-binary) samples: {int(len(metadata) - len(labels))}")
            
            return labels
        else:
            # 使用Group列
            unique_groups = metadata[group_col].unique()
            if len(unique_groups) != 2:
                raise ValueError(f"Expected 2 groups, found {len(unique_groups)}: {unique_groups}")
            
            # 如果用户指定了label_0和label_1，使用指定的映射
            if label_0 is not None and label_1 is not None:
                if label_0 not in unique_groups or label_1 not in unique_groups:
                    raise ValueError(f"Specified labels not found in data. Available: {unique_groups}, Specified: 0='{label_0}', 1='{label_1}'")
                group_mapping = {label_0: 0, label_1: 1}
                self.logger.info(f"Using specified label mapping: 0='{label_0}', 1='{label_1}'")
            else:
                # 默认按字母顺序排序（保持向后兼容）
                sorted_groups = sorted(unique_groups)
                group_mapping = {sorted_groups[0]: 0, sorted_groups[1]: 1}
                self.logger.info(f"Using default label mapping: 0='{sorted_groups[0]}', 1='{sorted_groups[1]}'")
            
            labels = metadata[group_col].map(group_mapping).values
            
            # 存储标签映射信息供后续使用
            self.label_mapping_ = group_mapping
            
            return labels
    
    def _generate_groups(self, metadata: pd.DataFrame, cohort_column: str) -> np.ndarray:
        """Generate group information for cohort analysis."""
        return metadata[cohort_column].values
    
    def _process_data(self, data: pd.DataFrame, use_presence_absence: bool, use_clr: bool) -> pd.DataFrame:
        """Process data based on parameters."""
        if use_presence_absence:
            # 转换为有无数据 (0/1) - 使用与原始项目相同的阈值
            processed_data = data.applymap(lambda x: 1 if x > 1e-4 else 0)
            self.logger.info("Data transformation: Presence/absence (threshold=1e-4)")
        else:
            # 保持原始相对丰度数据
            processed_data = data.copy()
            if use_clr:
                # 应用CLR变换
                processed_data = self._apply_clr_transform(processed_data)
                self.logger.info("Data transformation: CLR-transformed relative abundance")
            else:
                self.logger.info("Data transformation: Original relative abundance")
        
        # 移除常数特征
        processed_data = self._remove_constant_features(processed_data)
        
        return processed_data
    
    def _process_data_with_tracking(self, data: pd.DataFrame, use_presence_absence: bool, use_clr: bool) -> Tuple[pd.DataFrame, List[str]]:
        """Process data with tracking of removed features."""
        if use_presence_absence:
            # 转换为有无数据 (0/1) - 使用与原始项目相同的阈值
            processed_data = data.applymap(lambda x: 1 if x > 1e-4 else 0)
            self.logger.info("Data transformation: Presence/absence (threshold=1e-4)")
        else:
            # 保持原始相对丰度数据
            processed_data = data.copy()
            if use_clr:
                # 应用CLR变换
                processed_data = self._apply_clr_transform(processed_data)
                self.logger.info("Data transformation: CLR-transformed relative abundance")
            else:
                self.logger.info("Data transformation: Original relative abundance")
        
        # 移除常数特征并记录被移除的特征
        processed_data, constant_removed_features = self._remove_constant_features_with_tracking(processed_data)
        
        return processed_data, constant_removed_features
    
    def _apply_clr_transform(self, data: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
        """
        Apply Centered Log-Ratio (CLR) transformation to relative abundance data.
        
        CLR transformation is commonly used for compositional data like microbiome data.
        It transforms relative abundances to log-ratios, making the data more suitable
        for standard statistical methods.
        
        Args:
            data: Relative abundance data (samples x features)
            pseudocount: Small value added to avoid log(0)
            
        Returns:
            CLR-transformed data with same index and columns as input
        """
        import numpy as np
        
        self.logger.info(f"Applying CLR transformation to {data.shape[0]} samples x {data.shape[1]} features")
        self.logger.info(f"Data range before CLR: [{data.min().min():.6f}, {data.max().max():.6f}]")
        
        # Add pseudocount to avoid log(0)
        data_with_pseudo = data + pseudocount
        
        # Calculate geometric mean for each sample (row-wise)
        # For each sample, compute geometric mean of all species abundances
        geometric_mean = data_with_pseudo.apply(lambda row: np.exp(np.mean(np.log(row))), axis=1)
        
        # Apply CLR transformation: log(x / geometric_mean)
        # This is done element-wise: each abundance divided by its sample's geometric mean
        clr_data = data_with_pseudo.div(geometric_mean, axis=0).apply(np.log)
        
        # Handle any remaining NaN or inf values
        clr_data = clr_data.replace([np.inf, -np.inf], np.nan)
        if clr_data.isnull().any().any():
            self.logger.warning("CLR transformation produced NaN values, filling with 0")
            clr_data = clr_data.fillna(0)
        
        self.logger.info(f"Data range after CLR: [{clr_data.min().min():.6f}, {clr_data.max().max():.6f}]")
        
        return clr_data
    
    def _remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        移除常数特征（全为0或全为1的特征）
        
        Args:
            X: 特征矩阵
            
        Returns:
            过滤后的特征矩阵
        """
        original_shape = X.shape
        constant_features = []

        # 若无列，直接返回（避免后续访问）
        if original_shape[1] == 0:
            self.logger.info("未发现特征列，跳过常量过滤")
            return X

        for col in X.columns:
            unique_values = X[col].nunique()
            if unique_values <= 1:  # 只有0个或1个唯一值
                constant_features.append(col)

        if constant_features:
            X_filtered = X.drop(columns=constant_features)
            self.logger.info(f"移除常数特征: {len(constant_features)} 个特征被移除")
            self.logger.info(f"  原始特征数: {original_shape[1]}")
            self.logger.info(f"  过滤后特征数: {X_filtered.shape[1]}")
            self.logger.info(f"  移除比例: {len(constant_features)/original_shape[1]:.1%}")
            return X_filtered
        else:
            self.logger.info("未发现常数特征")
            return X
    
    def _remove_constant_features_with_tracking(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        移除常数特征（全为0或全为1的特征）并返回被移除的特征列表
        
        Args:
            X: 特征矩阵
            
        Returns:
            Tuple of (过滤后的特征矩阵, 被移除的特征列表)
        """
        original_shape = X.shape
        constant_features = []

        # 若无列，直接返回（避免后续访问）
        if original_shape[1] == 0:
            self.logger.info("未发现特征列，跳过常量过滤")
            return X, []

        for col in X.columns:
            unique_values = X[col].nunique()
            if unique_values <= 1:  # 只有0个或1个唯一值
                constant_features.append(col)

        if constant_features:
            X_filtered = X.drop(columns=constant_features)
            self.logger.info(f"移除常数特征: {len(constant_features)} 个特征被移除")
            self.logger.info(f"  原始特征数: {original_shape[1]}")
            self.logger.info(f"  过滤后特征数: {X_filtered.shape[1]}")
            self.logger.info(f"  移除比例: {len(constant_features)/original_shape[1]:.1%}")
            return X_filtered, constant_features
        else:
            self.logger.info("未发现常数特征")
            return X, []
    
    def _print_data_info(self, X: pd.DataFrame, cohort_info: Optional[np.ndarray]) -> None:
        """Print basic info and preview of data; no return (user feedback only)."""
        self.logger.info("----------------------------")
        self.logger.info("Data Information:")
        self.logger.info(f"Data shape: {X.shape}")
        self.logger.info(f"First 5 rows and 5 columns:")
        self.logger.info(f"{X.iloc[:5, :5]}")
        
        if cohort_info is not None:
            self.logger.info(f"Unique cohort values: {np.unique(cohort_info)}")
        else:
            self.logger.info("LOCO not applicable, cohort_info is None")
        self.logger.info("----------------------------")
    
    def load_from_config(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Load data from configuration dictionary."""
        return self.load_microbiome_data(
            profile_path=config['profile_path'],
            metadata_path=config['metadata_path'],
            project=config.get('project'),
            sample_id_col=config.get('sample_id_col', 'SampleID'),
            project_col=config.get('project_col', 'Project'),
            group_col=config.get('group_col', 'Group')
        )
