#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"    // must include this file in order to use NCNN net.

int main(int argc, char** argv) {
    // Load image
    std::string imagepath("../wuxiangzhileisq.png");
    // IMREAD_COLOR: always convert image to the 3 channel BGR color image.
    cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Unable to read image file " << imagepath << std::endl;
        return -1;
    }
    std::cout << "Image processed" << std::endl;
    
//     // Image preprocessing
//     img.convertTo(img, CV_32F);
//     double max_val;
//     cv::minMaxLoc(img, nullptr, &max_val);
//     int max_range = 255;
//     if (max_val > 256) {
//         max_range = 65535;
//         std::cout << "\tInput is a 16-bit image" << std::endl;
//     }
//     img /= max_range;
    
    // Load NCNN model
    ncnn::Net net;
    int ret = net.load_param("../RealESRGAN_x4plus_anime_6B_opt.param");
    if (ret) {
        std::cerr << "Failed to load model parameters " << ret << std::endl;
        return -1;
    }
    std::cout << "Param loaded" << std::endl;
    ret = net.load_model("../RealESRGAN_x4plus_anime_6B_opt.bin");
    if (ret) {
        std::cerr << "Failed to load model weights " << ret << std::endl;
        return -1;
    }
    std::cout << "Bin loaded" << std::endl;

    // Convert image data to ncnn format
    // opencv image in bgr, model needs bgr
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, 
        ncnn::Mat::PIXEL_BGR, img.cols, img.rows);

    // Inference
    ncnn::Extractor extractor = net.create_extractor();
    extractor.input("in0", input);
    ncnn::Mat output;
    extractor.extract("out0", output);

    // Image postprocessing
    cv::Mat a(output.h, output.w, CV_8UC3);
    output.to_pixels(a.data, ncnn::Mat::PIXEL_BGR);
//     cv::resize(a, a, cv::Size(int(img.cols * 1), int(img.rows * 1)), 0, 0, cv::INTER_LANCZOS4);
//     if (max_range == 65535) {
//         a.convertTo(a, CV_16U, 65535.0);
//     } else {
//         a.convertTo(a, CV_8U, 255.0);
//     }
    cv::imwrite("../wuxiangzhileisq_RealESRGAN_x4plus_anime_6B.png", a);
    std::cout << "Completed" << std::endl;
    return 0;
}