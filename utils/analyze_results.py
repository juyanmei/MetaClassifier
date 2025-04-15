import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
from sklearn.metrics import roc_curve, auc


def generate_boxplots(data_file, args, directory):
    # 读取数据
    #df = pd.read_csv(data_file, sep=',', index_col=0)
    df = data_file
    group_columns = ['target', 'target_id', 'feature', 'turner']
    df_cleaned = df.drop(columns=group_columns)
    model_column = 'Model'
    # 处理缺失值
    df_cleaned = df_cleaned.dropna()

    tmp_target_id = getattr(args, 'target_id', 'merge')
    pdf_filename = f"{directory}/{args.target}_{tmp_target_id}_Scores_Feature_{args.feature}_Turner{args.turner}.pdf"

    with PdfPages(pdf_filename) as pdf:
        # 对每个数值列绘制箱线图
        for column in df_cleaned.columns:
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                fig, ax = plt.subplots(figsize=(10, 6))
                df.boxplot(column=column, by=model_column, ax=ax)
                medians = df.groupby(model_column)[column].median()
                for i, (model, median) in enumerate(medians.items()):
                    ax.text(i + 1, median, f'{median:.2f}', ha='center', va='bottom', color='black', fontsize=10)
                # Set plot title and labels
                ax.set_title(f'Boxplot for {column}')
                ax.set_xlabel(model_column)
                ax.set_ylabel(column)
                ax.grid(False)
                pdf.savefig(fig)
                plt.close()
    print(f"All boxplots have been saved to '{pdf_filename}'.")

def generate_heatmap(df, args, directory):
    """
    生成数据框的热图并保存到指定的PDF目录
    :param df: 输入的数据框
    :param directory: 输出目录
    """
    # 定义颜色映射
    colors = [(0, "white"), (0.5, "#88CCEE"), (0.7, "#DDCC77"), (1, "#CC6677")]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # 定义PDF文件路径
    pdf_path = f"{directory}/{args.target}_Heatmap_Feature_{args.feature}_Turner{args.turner}.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".2f", cbar_kws={'ticks': [0, 0.5, 0.7, 1]})
        plt.title("Custom Heatmap with Values")
        
        # Save plot to PDF
        pdf.savefig()
        plt.close()
    
    print(f"Heatmap has been saved to '{pdf_path}'.")

def find_optimal_cutoff(y_true, y_pred):
    """
    计算基于ROC曲线的最优cutoff值
    :param y_true: 真实标签
    :param y_pred: 预测值
    :return: 最优cutoff值
    """
    # 确保 y_true 是二进制标签
    if not set(y_true).issubset({0, 1}):
        raise ValueError("y_true should only contain binary labels (0 and 1).")
    # 确保 y_pred 是浮点数
    if not np.issubdtype(y_pred.dtype, np.floating):
        raise ValueError("y_pred should be of float type.")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    J = tpr - fpr
    ix = np.argmax(J)
    best_threshold = thresholds[ix]
    return best_threshold, fpr, tpr, auc(fpr, tpr)


def analyze_predictions(data_file, args, directory):
    """
    分析模型预测值，计算最优cutoff值，绘制ROC曲线和Index箱线图，并将结果保存到指定目录
    :param data_file: 数据文件路径
    :param directory: 输出目录
    """
    # 读取数据
    #df = pd.read_csv(data_file, sep=',', index_col=0)
    df = data_file
    y_true = df['true_values']
    columns_to_drop = ['true_values', 'target', 'target_id', 'feature', 'turner']
    df_cleaned = df.drop(columns=columns_to_drop)
    model_columns = df_cleaned.columns

    # 创建一个新的DataFrame来存储Index值
    index_df = pd.DataFrame()
    cutoff_values = []
    tmp_target_id = getattr(args, 'target_id', 'merge')
    pdf_filename = f"{directory}/{args.target}_{tmp_target_id}_ROC_Feature{args.feature}_Turner{args.turner}.pdf" 
    # 计算每个模型的最优cutoff值并绘制ROC曲线
    with PdfPages(pdf_filename) as pdf:
        plt.figure()
        for model_column in model_columns:
            y_pred = df[model_column]
            optimal_cutoff, fpr, tpr, roc_auc = find_optimal_cutoff(y_true, y_pred)
            print(f"The optimal cutoff value for {model_column} is: {optimal_cutoff}")
            print(f"The AUC value for {model_column} is: {roc_auc}")
            # 存储cutoff值
            cutoff_values.append([model_column, optimal_cutoff])
            # 计算Index值
            index_values = optimal_cutoff - y_pred
            index_df[f'{model_column}_index'] = index_values
            # 绘制ROC曲线
            plt.plot(fpr, tpr, lw=2, label=f'{model_column} (AUC = {roc_auc:.2f})')
        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # Set plot title and labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        # 保存ROC曲线图像到PDF
        pdf.savefig()
        plt.close()
    
    # 将真实值添加到index_df中
    index_df['true_label_column'] = y_true
    # 绘制Index箱线图
    index_boxplot_path = f"{directory}/{args.target}_{tmp_target_id}_IndexBoxplot_Feature{args.feature}_Turner{args.turner}.pdf"
    with PdfPages(index_boxplot_path) as pdf:
        for model_column in model_columns:
            index_column = f'{model_column}_index'
            fig, ax = plt.subplots(figsize=(10, 6))
            index_df.boxplot(column=index_column, by='true_label_column', ax=ax)
            
            # Set plot title and labels
            ax.set_title(f'Index Boxplot for {model_column}')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Index')
            ax.grid(False)
            # Save plot to PDF
            pdf.savefig(fig)
            plt.close()
    
    print(f"All boxplots have been saved to '{index_boxplot_path}'.")
    
    # 将cutoff值写入到本地文件
    cutoff_df = pd.DataFrame(cutoff_values, columns=['Model', 'Cutoff'])
    cutoff_file_path = f"{directory}/{args.target}_{tmp_target_id}_Cutoff_Feature{args.feature}_Turner{args.turner}.csv"
    cutoff_df.to_csv(cutoff_file_path, index=True)
    print(f"Cutoff values have been saved to '{cutoff_file_path}'.")