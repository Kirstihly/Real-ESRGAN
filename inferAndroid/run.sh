rm RealESRGAN_x4plus_anime_6B* && rm debug* && rm -rf build

python pt2onnx.py --params

pnnx RealESRGAN_x4plus_anime_6B.onnx

/Documents/ncnn/build/tools/ncnnoptimize RealESRGAN_x4plus_anime_6B.ncnn.param RealESRGAN_x4plus_anime_6B.ncnn.bin RealESRGAN_x4plus_anime_6B_opt.param RealESRGAN_x4plus_anime_6B_opt.bin 65536

mkdir build && cd build
cmake ..
make
./esrgan_anime
