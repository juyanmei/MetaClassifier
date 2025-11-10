#!/usr/bin/env python3
"""
metaClassifier v1.0 - 宏基因组分类模型构建终极指南

基于两阶段架构的完整机器学习流水线：
1. 第一阶段：嵌套CV评估（无偏性能估计 + 共识特征选择）
2. 第二阶段：最终模型训练（使用共识特征集 + 超参数调优）

支持两种模式：
- build: 构建和训练模型
- report: 生成分析报告
"""

import sys
import os
import warnings
import numpy as np
import random
import logging
from datetime import datetime
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    # 如果 sklearn 不可用或异常，忽略过滤设置
    pass
from pathlib import Path

# 设置全局随机种子以确保可重复性
def set_global_seed(seed: int = 42):
    """设置所有随机种子以确保可重复性"""
    # 设置Python内置随机种子
    random.seed(seed)
    
    # 设置numpy随机种子
    np.random.seed(seed)
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 用于CUDA确定性
    
    # 设置其他可能的随机源
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # 设置sklearn随机种子
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    # 设置pandas随机种子（如果可用）
    try:
        import pandas as pd
        # pandas本身没有全局随机种子，但我们可以确保某些操作是确定性的
        pd.set_option('mode.chained_assignment', None)
    except ImportError:
        pass

# 在程序开始时设置全局随机种子
set_global_seed(42)

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from metaClassifier.cli.argument_parser import parse_arguments


class TeeLogger:
    """将输出同时写入到终端和日志文件"""
    
    def __init__(self, log_file_path: str = None):
        self.terminal = sys.stdout
        self.log_file = None
        self.log_file_path = log_file_path
        if log_file_path:
            self._open_log_file(log_file_path)
    
    def _open_log_file(self, log_file_path: str):
        """打开日志文件"""
        # 确保目录存在
        from pathlib import Path
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_file_path = log_file_path
    
    def set_log_file(self, log_file_path: str):
        """动态设置日志文件路径"""
        # 关闭旧的日志文件
        if self.log_file:
            self.log_file.close()
        # 打开新的日志文件
        self._open_log_file(log_file_path)
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # 添加其他必要的方法以完全模拟stdout
    def __getattr__(self, name):
        return getattr(self.terminal, name)


def setup_output_logging(args):
    """设置输出日志记录"""
    # 日志文件路径将在build pipeline中动态设置
    # 这里不需要提前确定路径
    return None


def main():
    """主入口函数，根据命令分发到相应的处理器。"""
    args = parse_arguments()
    
    # 命令处理器映射
    handlers = {
        'build': lambda a: __import__('metaClassifier.pipelines.build', fromlist=['handle_build']).handle_build(a),
        'report': lambda a: __import__('metaClassifier.pipelines.report', fromlist=['handle_report']).handle_report(a),
    }
    
    cmd = getattr(args, 'command', None)
    handler = handlers.get(cmd)
    
    if handler is None:
        raise ValueError(f"Unknown command: {cmd}. Supported commands: {', '.join(handlers.keys())}")
    
    # 设置输出日志记录（初始为None，将在handler中通过环境变量设置）
    log_file_path = None
    
    # 使用TeeLogger记录输出
    # 注意：初始时log_file_path为None，TeeLogger只输出到终端
    # handler执行后，会通过环境变量METACLASSIFIER_LOG_FILE设置日志文件路径
    with TeeLogger(log_file_path) as tee_logger:
        # 重定向stdout到TeeLogger
        original_stdout = sys.stdout
        sys.stdout = tee_logger
        
        # 设置logger也输出到TeeLogger
        import logging
        original_logger_handlers = []
        if log_file_path:
            # 获取根logger
            root_logger = logging.getLogger()
            # 保存原始handlers
            original_logger_handlers = root_logger.handlers[:]
            # 清除现有handlers
            root_logger.handlers.clear()
            
            # 添加TeeLogger作为handler
            class TeeHandler(logging.StreamHandler):
                def __init__(self, tee_logger):
                    super().__init__()
                    self.tee_logger = tee_logger
                
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        self.tee_logger.write(msg + '\n')
                        self.tee_logger.flush()
                    except Exception:
                        self.handleError(record)
                
                def flush(self):
                    if hasattr(self.tee_logger, 'flush'):
                        self.tee_logger.flush()
            
            tee_handler = TeeHandler(tee_logger)
            tee_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
            tee_handler.setFormatter(formatter)
            root_logger.addHandler(tee_handler)
            root_logger.setLevel(logging.INFO)
            
            # 关键修复：重写get_logger函数，确保所有新创建的logger都使用TeeLogger
            from metaClassifier.utils.logger import get_logger as original_get_logger
            
            def patched_get_logger(name: str, level: int = logging.INFO):
                logger = logging.getLogger(name)
                if not logger.handlers:
                    # 创建TeeLogger handler
                    tee_handler = TeeHandler(tee_logger)
                    tee_handler.setLevel(level)
                    formatter = logging.Formatter(
                        '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                    tee_handler.setFormatter(formatter)
                    logger.addHandler(tee_handler)
                    logger.setLevel(level)
                    logger.propagate = True  # 关键：设置为True，让日志传播到根logger
                return logger
            
            # 替换get_logger函数
            import metaClassifier.utils.logger
            metaClassifier.utils.logger.get_logger = patched_get_logger
        
        try:
            # 记录开始时间
            start_time = datetime.now()
            sys.stdout.write(f"================================================================================\n"
                           f"MetaClassifier 运行日志\n"
                           f"================================================================================\n"
                           f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"命令: {cmd.upper()}\n"
                           f"工作目录: {os.getcwd()}\n"
                           f"Python版本: {sys.version}\n"
                           f"================================================================================\n")
            sys.stdout.flush()
            
            # 执行handler
            handler(args)
            
            sys.stdout.write(f"\n✅ {cmd.upper()} 命令执行完成！\n")
            
            # 记录结束时间
            end_time = datetime.now()
            duration = end_time - start_time
            sys.stdout.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"总耗时: {duration}\n"
                           f"================================================================================\n")
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            sys.stdout.write(f"\n⚠️ {cmd.upper()} 命令被用户中断\n")
            sys.exit(130)
        except FileNotFoundError as e:
            sys.stdout.write(f"\n❌ 文件未找到: {e}\n")
            sys.exit(2)
        except ValueError as e:
            sys.stdout.write(f"\n❌ 参数错误: {e}\n")
            sys.exit(3)
        except ImportError as e:
            sys.stdout.write(f"\n❌ 导入错误: {e}\n")
            sys.stdout.write("请检查依赖包是否正确安装\n")
            sys.exit(4)
        except Exception as e:
            sys.stdout.write(f"\n❌ {cmd.upper()} 命令执行失败: {e}\n")
            # 如果启用了详细模式，打印完整错误信息
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                sys.stdout.write("\n详细错误信息:\n")
                traceback.print_exc()
            sys.exit(1)
        finally:
            # 恢复原始stdout和logger
            sys.stdout = original_stdout
            if log_file_path and original_logger_handlers:
                root_logger = logging.getLogger()
                root_logger.handlers.clear()
                root_logger.handlers.extend(original_logger_handlers)


if __name__ == "__main__":
    main()
