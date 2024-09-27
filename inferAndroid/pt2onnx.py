# import sys
# sys.path.append("../Real-ESRGAN/")

import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer

print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.cuda.current_device()", torch.cuda.current_device())
print("torch.cuda.device(0)", torch.cuda.device(0))
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

def main(args):
    # An instance of the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    if args.params:
        print("keyname: params")
        keyname = 'params'
    else:
        print("keyname: params_ema")
        keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()
#     model.eval()

#     upsampler = RealESRGANer(
#         scale=4,
#         model_path=args.input,
#         dni_weight=None,
#         model=model,
#         tile=0,
#         tile_pad=10,
#         pre_pad=0,
#         half=True,
#         gpu_id=None)

    # An example input
    x = torch.rand(1, 3, 128, 128)
    # Export the model
    with torch.no_grad():
        torch.onnx.export(model, x, args.output, opset_version=11, export_params=True, input_names=['input'], output_names=['output'])
    print("Saved:", args.output)

if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='/Documents/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='RealESRGAN_x4plus_anime_6B.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    
    args = parser.parse_args()

    main(args)