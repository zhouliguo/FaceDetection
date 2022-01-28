#include <highgui.hpp>
#include <imgproc.hpp>
#include <core.hpp>
#include <dnn.hpp>
#include <fstream>
#include <ctime>
#include <iostream>


int main() {
	std::string path;

	std::ifstream val_list("image_list_val.txt");

	cv::dnn::Net net = cv::dnn::readNetFromONNX("D:/vs/FaceDetection/weights/weight_light_640.onnx");

	double time_sum = 0;
	clock_t start, end;
	for (int i = 0; i < 3226; i++) {
		val_list >> path;
		cv::Mat image = cv::imread("D:/WIDER_FACE/WIDER_val/images/" + path);
		cv::resize(image, image, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
		cv::Mat blob = cv::dnn::blobFromImage(image);
		net.setInput(blob);
		start = clock();		//程序开始计时
		cv::Mat predict = net.forward();
		end = clock();		//程序结束用时
		time_sum = time_sum + (double)(end - start);
	}
	std::cout << "Total time:" << time_sum << "ms" << std::endl;	//ms为单位
	std::cout << "FPS:" << 3226 * 1000 / time_sum << "ms" << std::endl;	//ms为单位

	return 0;
}
