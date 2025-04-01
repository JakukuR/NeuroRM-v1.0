// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/inference.h"

// static constexpr int INPUT_W = 640;    // Width of input
// static constexpr int INPUT_H = 384;    // Height of input
static constexpr int INPUT_W = 416;    // Width of input
static constexpr int INPUT_H = 416;    // Height of input
static constexpr int NUM_CLASSES = 9;  // Number of classes
static constexpr int NUM_COLORS = 4;   // Number of color
static constexpr int TOPK = 128;       // TopK
static constexpr float NMS_THRESH = 0.3;
static constexpr float BBOX_CONF_THRESH = 0.6;
static constexpr float FFT_CONF_ERROR = 0.15;
static constexpr float FFT_MIN_IOU = 0.9;

static inline int argmax(const float *ptr, int len)
{
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

/**
 * @brief Resize the image using letterbox
 * @param img Image before resize
 * @param transform_matrix Transform Matrix of Resize
 * @return Image after resize
 */
inline cv::Mat scaledResize(cv::Mat& img, Eigen::Matrix<float,3,3> &transform_matrix)
{
    float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;

    dw /= 2;
    dh /= 2;

    transform_matrix << 1.0 / r, 0, -dw / r,
            0, 1.0 / r, -dh / r,
            0, 0, 1;

    Mat re;
    cv::resize(img, re, Size(unpad_w,unpad_h));
    Mat out;
    cv::copyMakeBorder(re, out, dh, dh, dw, dw, BORDER_CONSTANT);

    return out;
}

/**
 * @brief Generate grids and stride.
 * @param target_w Width of input.
 * @param target_h Height of input.
 * @param strides A vector of stride.
 * @param grid_strides Grid stride generated in this function.
 */
static void generate_grids_and_stride(
    const int target_w, const int target_h,
    std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;

        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

/**
 * @brief Generate Proposal
 * @param grid_strides Grid strides
 * @param feat_ptr Original predition result.
 * @param prob_threshold Confidence Threshold.
 * @param objects Objects proposed.
 */
static void generateYoloxProposals(
    std::vector<GridAndStride> grid_strides,
    const float* feat_ptr,
    Eigen::Matrix<float,3,3> &transform_matrix,
    float prob_threshold,
    std::vector<ArmorObject>& objects)
{

    const int num_anchors = grid_strides.size();
    //Travel all the anchors
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (9 + NUM_COLORS + NUM_CLASSES);

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
        float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
        float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
        float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
        float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
        float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
        float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
        float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;

        int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        float box_objectness = (feat_ptr[basic_pos + 8]);

        float color_conf = (feat_ptr[basic_pos + 9 + box_color]);
        float cls_conf = (feat_ptr[basic_pos + 9 + NUM_COLORS + box_class]);

        // float box_prob = (box_objectness + cls_conf + color_conf) / 3.0;
        float box_prob = box_objectness;

        if (box_prob >= prob_threshold)
        {
            ArmorObject obj;

            Eigen::Matrix<float,3,4> apex_norm;
            Eigen::Matrix<float,3,4> apex_dst;

            apex_norm << x_1,x_2,x_3,x_4,
                    y_1,y_2,y_3,y_4,
                    1,1,1,1;

            apex_dst = transform_matrix * apex_norm;

            for (int i = 0; i < 4; i++)
            {
                obj.apex[i] = cv::Point2f(apex_dst(0,i),apex_dst(1,i));
                obj.pts.push_back(obj.apex[i]);
            }

            vector<cv::Point2f> tmp(obj.apex,obj.apex + 4);
            obj.rect = cv::boundingRect(tmp);

            obj.cls = box_class;
            obj.color = box_color;
            obj.prob = box_prob;

            objects.push_back(obj);
        }

    } // point anchor loop
}

/**
 * @brief Calculate intersection area between two objects.
 * @param a Object a.
 * @param b Object b.
 * @return Area of intersection.
 */
static inline float intersection_area(const ArmorObject& a, const ArmorObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<ArmorObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}


static void qsort_descent_inplace(std::vector<ArmorObject>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}


static void nms_sorted_bboxes(std::vector<ArmorObject>& faceobjects, std::vector<int>& picked,
                              float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        ArmorObject& a = faceobjects[i];

        int keep = 1;
        for (int j : picked)
        {
            ArmorObject& b = faceobjects[j];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold)
            {
                keep = 0;
                //Stored for FFT
                if (iou > FFT_MIN_IOU && abs(a.prob - b.prob) < FFT_CONF_ERROR
                    && a.cls == b.cls && a.color == b.color)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        b.pts.push_back(a.apex[i]);
                    }
                }
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

/**
 * @brief Decode outputs.
 * @param prob Original predition output.
 * @param objects Vector of objects predicted.
 * @param img_w Width of Image.
 * @param img_h Height of Image.
 */
static void decodeOutputs(const float* prob, std::vector<ArmorObject>& objects,
                          Eigen::Matrix<float,3,3> &transform_matrix, const int img_w, const int img_h)
{
    std::vector<ArmorObject> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
    generateYoloxProposals(grid_strides, prob, transform_matrix, BBOX_CONF_THRESH, proposals);
    qsort_descent_inplace(proposals);

    if (proposals.size() >= TOPK)
        proposals.resize(TOPK);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }
}

ArmorDetector::ArmorDetector()
= default;

ArmorDetector::~ArmorDetector()
= default;

//TODO:change to your dir
bool ArmorDetector::initModel(const string& path)
{
    //---------------载入并编译模型--------------//
    compiled_model = ie.compile_model(path,"GPU");//choose CPU or GPU
    //---------------创建推理请求----------------//
    infer_request = compiled_model.create_infer_request();
    //----------------设置模型输入--------------//
    input_tensor=infer_request.get_input_tensor();
    //--------------获取输入shape---------------//
    tensor_shape=input_tensor.get_shape();
    return true;
}

bool ArmorDetector::detect(Mat &src,std::vector<ArmorObject>& objects)
{
    if (src.empty())
    {
        std::cout << " ERROR: 传入了空的src " << std::endl;
        return false;
    }
    cv::Mat pr_img = scaledResize(src,transfrom_matrix);
#ifdef SHOW_INPUT
    namedWindow("network_input",0);
    imshow("network_input",pr_img);
    waitKey(1);
#endif //SHOW_INPUT
    cv::Mat pre;
    cv::Mat pre_split[3];
    pr_img.convertTo(pre,CV_32F);
    cv::split(pre,pre_split);

    float* image_data = (input_tensor.data<float>());
    auto img_offset = INPUT_W * INPUT_H;
    for(auto & c : pre_split)
    {
        memcpy(image_data, c.data, INPUT_W * INPUT_H * sizeof(float));
        image_data += img_offset;
    }

    auto t1 = std::chrono::steady_clock::now();
    infer_request.infer();
    auto t2 = std::chrono::steady_clock::now();
    cout<<(float)(std::chrono::duration<double,std::milli>(t2 - t1).count())<<endl;

    auto output_tensor = infer_request.get_output_tensor();
    Shape out_shape = output_tensor.get_shape();
    int img_w = src.cols;
    int img_h = src.rows;
    auto net_pred = output_tensor.data<float>();;

    decodeOutputs(net_pred, objects, transfrom_matrix, img_w, img_h);
    for (auto & object : objects)
    {
        //对候选框预测角点进行平均,降低误差
        if (object.pts.size() >= 8)
        {
            auto N = object.pts.size();
            cv::Point2f pts_final[4];

            for (int i = 0; i < N; i++)
            {
                pts_final[i % 4]+=object.pts[i];
            }

            for (auto & i : pts_final)
            {
                i.x = i.x / (N / 4);
                i.y = i.y / (N / 4);
            }

            object.apex[0] = pts_final[0];
            object.apex[1] = pts_final[1];
            object.apex[2] = pts_final[2];
            object.apex[3] = pts_final[3];

        }
        object.area = (int)(calcTetragonArea(object.apex));
    }
    if (objects.size() != 0)
        return true;
    else
        return false;
}