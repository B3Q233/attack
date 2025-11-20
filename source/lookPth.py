import torch
import csv
import os
from collections import OrderedDict
from typing import OrderedDict as OdType, Union, List
import numpy as np


def get_model_weights(net: OdType, effective_num: int) -> OdType:
    """
    从模型状态字典中读取权重，保留指定小数位数并返回OrderedDict
    
    Args:
        net: 模型状态字典（包含参数名称和权重张量）
        effective_num: 保留的小数位数
        
    Returns:
        处理后的权重字典
    """
    if not isinstance(net, OrderedDict):
        raise TypeError("输入必须是OrderedDict类型的模型状态字典")
    if not isinstance(effective_num, int) or effective_num < 0:
        raise ValueError("小数位数必须是非负整数")

    weights = OrderedDict()
    scale = 10 **effective_num  # 缩放因子
    
    for name, param in net.items():
        # 确保参数是张量类型
        if not isinstance(param, torch.Tensor):
            raise TypeError(f"参数 {name} 不是torch.Tensor类型")
        
        # 四舍五入处理并保留原数据类型
        rounded_param = torch.round(param.data * scale) / scale
        weights[name] = rounded_param.to(param.dtype)  # 保持数据类型一致
    
    return weights


def save_model_params_to_csv(
    weights: OdType,
    csv_file_path: str,
    flatten: bool = True
) -> None:
    """
    将模型权重保存到CSV文件
    
    Args:
        weights: 处理后的模型权重字典
        csv_file_path: 保存CSV的路径
        flatten: 是否将多维张量展平为一维
    """
    if not weights:
        raise ValueError("权重字典为空，无法保存")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['参数名称', '权重值'])
        
        for name, param in weights.items():
            # 转换为numpy数组处理
            param_np = param.cpu().numpy()
            
            # 展平处理（如果需要）
            if flatten and param_np.ndim > 1:
                param_flat = param_np.flatten()
                # 多维参数分多行保存（每行一个元素）
                for idx, val in enumerate(param_flat):
                    writer.writerow([f"{name}[{idx}]", val])
            else:
                # 标量或一维张量直接保存
                writer.writerow([name, param_np.item() if param_np.ndim == 0 else param_np.tolist()])
    
    print(f"权重已成功保存到: {os.path.abspath(csv_file_path)}")


if __name__ == '__main__':
    # 配置参数
    pth_file = r'code\checkpoints\lgn-ML100k-3-150.pth.tar'  # 使用单斜杠更规范
    csv_save_path = r'code\checkpoints\model_weights.csv'  # CSV保存路径
    decimal_places = 2  # 保留两位小数
    flatten_params = True  # 展平多维参数
    
    try:
        # 加载模型权重
        if not os.path.exists(pth_file):
            raise FileNotFoundError(f"模型文件不存在: {pth_file}")
        
        model = torch.load(pth_file, map_location=torch.device('cpu'))
        
        # 提取状态字典（兼容不同保存格式）
        if 'state_dict' in model:
            state_dict = model['state_dict']
        else:
            state_dict = model  # 直接保存的状态字典
        
        # 处理权重（保留指定小数位数）
        processed_weights = get_model_weights(state_dict, decimal_places)
        
        # 保存到CSV
        save_model_params_to_csv(processed_weights, csv_save_path, flatten_params)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")