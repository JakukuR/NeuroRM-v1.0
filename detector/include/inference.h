#ifndef YOLOXARMOR_INFERENCE_H
#define YOLOXARMOR_INFERENCE_H

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "general.h"

using namespace std;
using namespace cv;
using namespace ov;
//using namespace InferenceEngine;

struct ArmorObject
{
    Point2f apex[4];              // 灯条四点坐标
    cv::Rect_<float> rect;        // 灯条四点矩形
    int cls;                      // 类别 (0:哨兵 1:英雄 2：工程 3、4、5：步兵 6：前哨站 7：基地)
    int color;                    // 颜色分类 (0:蓝色 1:红色 2:灰色)
    int area;                     // 矩形面积大小
    float prob;                   // 分类置信度
    std::vector<cv::Point2f> pts; // 灯条四点坐标
};


class ArmorDetector
{
public:
    ArmorDetector();
    ~ArmorDetector();
    bool detect(Mat &src,vector<ArmorObject>& objects);
    bool initModel(const string& path);
private:

    Core ie;
    CompiledModel compiled_model;
    InferRequest infer_request;      // 推理请求
    Tensor input_tensor;
    Shape tensor_shape;

    string input_name;
    string output_name;

    Eigen::Matrix<float,3,3> transfrom_matrix;
};

#endif //YOLOXARMOR_INFERENCE_H
