"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.onnx
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_model_tahusyda import SSD300, MultiBoxLoss
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_utils_xslstyan import *

# python aicandy_ssd300_convert_rqicuvtl.py --model_path aicandy_model_out_gnloibxd/aicandy_checkpoint_ssd300.pth --onnx_path aicandy_model_out_gnloibxd/aicandy_model_kqgmngun.onnx

def convert_checkpoint_to_onnx(checkpoint_path, onnx_path, n_classes):
    # Tải checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Khởi tạo mô hình
    model = SSD300(n_classes=n_classes)
    
    # Tải trọng số từ checkpoint
    model.load_state_dict(checkpoint['model'].state_dict())
    
    # Đặt mô hình ở chế độ đánh giá
    model.eval()
    
    # Tạo đầu vào mẫu
    dummy_input = torch.randn(1, 3, 300, 300)
    
    # Xuất mô hình sang ONNX
    torch.onnx.export(model, dummy_input, onnx_path, 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['locations', 'class_scores'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'locations': {0: 'batch_size'},
                                    'class_scores': {0: 'batch_size'}})
    
    print(f"Mô hình đã được chuyển đổi và lưu tại {onnx_path}")

# Sử dụng hàm
if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model (.pth)')
    parser.add_argument('--onnx_path', type=str, default='model.onnx', help='Path to save the ONNX model')

    args = parser.parse_args()

    n_classes = len(label_map)  # Số lượng lớp, được định nghĩa trong utils.py    
    convert_checkpoint_to_onnx(args.model_path, args.onnx_path, n_classes)