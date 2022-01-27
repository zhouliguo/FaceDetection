#include <iostream>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <opencv.hpp>

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};


void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
        (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.height = (int)std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
    // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
    // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
    // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
}

void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::resize(image, resizedImage, cv::Size(640, 480));
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector< Detection> postprocessing(const cv::Size& resizedImageShape,
    const cv::Size& originalImageShape,
    std::vector<Ort::Value>& outputTensors,
    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

size_t vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

int main() {
	Ort::Env env{ nullptr };
	Ort::SessionOptions sessionOptions{ nullptr };
	Ort::Session session{ nullptr };

    const bool& isGPU = true;
    const std::wstring& modelPath = L"best.onnx";
    const std::string imagePath = "D:/WIDER_FACE/WIDER_train/images/0--Parade/0_Parade_marchingband_1_364.jpg";

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;
    const cv::Size& inputSize = cv::Size(640, 480);

    const float& confThreshold = 0.001;
    const float& iouThreshold = 0.6;

	env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
	sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    #ifdef _WIN32
        session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    #else
        session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    #endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1){
        std::cout << "Dynamic input shape" << std::endl;
        isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    inputNames.push_back(session.GetInputName(0, allocator));
    outputNames.push_back(session.GetOutputName(0, allocator));

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;

    inputImageShape = cv::Size2f(inputSize);

    cv::Mat image = cv::imread(imagePath);

    float* blob = nullptr;
    //std::vector<int64_t> inputTensorShape{ 1, 3, -1, -1 };
    inputTensorShape[0] = 1;
    preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));

    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), 1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);

    std::vector<Detection> results = postprocessing(resizedShape, image.size(), outputTensors, confThreshold, iouThreshold);

    for (Detection result : results) {
        cv::rectangle(image, result.box, cv::Scalar(0, 0, 255), 1, 1, 0);
        cv::imshow("image", image);
        cv::waitKey();
    }

	return 0;
}
