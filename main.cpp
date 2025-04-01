#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "inference.h"
#include <HaiKangCamera.h>


cv::Mat ori_src;
cv::Mat image2show;
VideoCapture capture;
ArmorDetector detector;

bool useROI = false;
cv::Rect roiRect;

void display(ArmorObject);

//***************** 打开海康相机 ******************//
HaiKangCamera HaiKang;
void Open_Haikang()
{
    HaiKang.StartDevice(0);
    // 设置分辨率
    HaiKang.SetResolution(1440,1080);
    HaiKang.SetFPS(400);
    //更新时间戳，设置时间戳偏移量
    //    HaiKang.UpdateTimestampOffset(time_start);
    // 开始采集帧
    HaiKang.SetStreamOn();
    // 设置曝光事件
    HaiKang.SetExposureTime(12000);
    // 设置1
    HaiKang.SetGAIN(0, 15);
    HaiKang.Set_Auto_BALANCE();
    HaiKang.GetImageParam(HaiKang.m_param);
    // SerialPort SP;
    // SendData send_data;
    // #ifdef OPEN_SERIAL
    //     while( bool usb = SP.initSerial("/dev/ttyUSB1", 921600, 'N', 8, 1) == false);
    //     send_data.clear();
    // #endif
}

int main() {
    // 选择视频源 (1、免驱相机  0、视频文件)
    int from_camera = 1;

    if (from_camera) {
        // capture.open(0);
        Open_Haikang();
    } else {
        string filename = "/home/jakukur/Downloads/yolox-openvino2022-master/videoTest/armor_red.avi";
        //string filename = "/home/jakukur/Downloads/3.MP4";
        capture.open(filename);
    }
    // if(!capture.isOpened()){
    //     printf("video can not open ...\n");
    //     return -1;
    // }

    // 初始化网络模型
    const string network_path = PROJECT_DIR"/model/yolox11.onnx";
    detector.initModel(network_path);

    while (true){

        if (from_camera) {
            // capture.read(ori_src); // 相机取图
            HaiKang.GetMat(ori_src);
            if (ori_src.empty()) { // 相机开启线程需要一定时间
                cerr << "无法显示图像\n";
                return -1;
            }
        }else {
            capture >> ori_src;
        }

        auto time_start=std::chrono::steady_clock::now();
        vector<ArmorObject> objects;  // 创建装甲板目标属性容器

        cv::Mat input;
        if (!useROI) {
            // 全图检测模式
            input = ori_src;  // 直接传入检测函数
        } else {
            // ROI检测模式，确保ROI在图像内
            cv::Rect validRoi = roiRect & cv::Rect(0, 0, ori_src.cols, ori_src.rows);
            //添加roi边界保护
            validRoi.x = max(0, validRoi.x);
            validRoi.y = max(0, validRoi.y);
            validRoi.width = max(0, min(ori_src.cols - validRoi.x, validRoi.width));
            validRoi.height = max(0, min(ori_src.rows - validRoi.y, validRoi.height));
            if (validRoi.width <= 0 || validRoi.height <= 0) {
                useROI = false;
                continue;  // 跳过本次检测
            }
            input = ori_src(validRoi);
        }

        image2show = ori_src; // 可视化图像
        // Mat input = ori_src;  // 网络推理图像

        bool found = detector.detect(input, objects);
        if (found && !objects.empty()){ // 前向推理获得目标结果
            for (auto armor_object : objects){
                display(armor_object); // 识别结果可视化
            }
            int targetCenterX, targetCenterY;
            if (useROI) {
                // ROI模式下，将检测结果转换为全图坐标
                targetCenterX = roiRect.x + objects[0].apex->x;
                targetCenterY = roiRect.y + objects[0].apex->y;
            } else {
                targetCenterX = objects[0].apex->x;
                targetCenterY = objects[0].apex->y;
            }
            // 构造以目标为中心、尺寸为640×480的ROI区域
            int roiWidth = 640, roiHeight = 640;
            roiRect = cv::Rect(targetCenterX - roiWidth/2, targetCenterY - roiHeight/2, roiWidth, roiHeight);

            int roi_x = std::max(0, targetCenterX - roiWidth/2);
            int roi_y = std::max(0, targetCenterY - roiHeight/2);
            int roi_w = std::min(ori_src.cols - roi_x, roiWidth);
            int roi_h = std::min(ori_src.rows - roi_y, roiHeight);
            roiRect = cv::Rect(roi_x, roi_y, roi_w, roi_h);
            // 切换到ROI检测模式
            // //判断roi是否超出图像边界
            // if (roiRect.x + roiRect.width > ori_src.cols || roiRect.y + roiRect.height > ori_src.rows) {
            //     useROI = true;
            // }

            useROI = true;

            // 可视化：绘制检测结果
            //display(detectedObjects[0]);
            // for (auto armor_object : objects){
            //     display(armor_object); // 识别结果可视化
            // }
        }else {
            if (useROI) {
                useROI = false;
            }
        }

        auto time_predict = std::chrono::steady_clock::now();
        double dr_full_ms = std::chrono::duration<double,std::milli>(time_predict - time_start).count();
        putText(image2show, "FPS: "+to_string(int(1000 / dr_full_ms)), {10, 25}, FONT_HERSHEY_SIMPLEX, 1, {0,255,0});
        cv::putText(image2show, "Latency:" + to_string(dr_full_ms) + "ms", Point(1080,25), FONT_HERSHEY_SIMPLEX, 1, {0,255,0});
        // cout <<"[AUTOAIM] LATENCY: "<< " Total: " << dr_full_ms << " ms"<< endl;

        namedWindow("output", WINDOW_NORMAL); // 创建窗口
        cv::resizeWindow("output", 1920, 1080);
        namedWindow("input", WINDOW_NORMAL);
        cv::resizeWindow("input", 1920, 1080);
        imshow("input", input);

        imshow("output", image2show);
        //按空格暫停或者繼續播放一下幀
        if (waitKey(1) == ' ') {
            while (waitKey(1) != ' ');
        }


    }
    return 0;
}

void display(ArmorObject object) {
    // 坐标转换：根据ROI状态调整坐标
    vector<Point2f> global_apex(4);
    for (int i = 0; i < 4; i++) {
        global_apex[i] = useROI
            ? Point2f(object.apex[i].x + roiRect.x, object.apex[i].y + roiRect.y)
            : object.apex[i];
    }

    // 绘制十字瞄准线
    line(image2show, Point2f(image2show.size().width / 2, 0), Point2f(image2show.size().width / 2, image2show.size().height), {0,255,0}, 1);
    line(image2show, Point2f(0, image2show.size().height / 2), Point2f(image2show.size().width, image2show.size().height / 2), {0,255,0}, 1);

    // 绘制装甲板四点矩形（使用全局坐标）
    for (int i = 0; i < 4; i++) {
        line(image2show, global_apex[i], global_apex[(i+1)%4], Scalar(100,200,0), 1);
        circle(image2show, global_apex[i], 3, Scalar(100,200,0), 1);
    }

    //  绘制四点
    // for (int i = 0; i < 4; i++) {
    //     circle(image2show, Point(object.apex[i].x + roiRect.x, object.apex[i].y + roiRect.y), 3, Scalar(100, 200, 0), 1);
    // }
    //  绘制左上角顶点
    //circle(image2show, Point(object.apex->x, object.apex->y),3,Scalar(255, 255, 0),1 );

    // // // 绘制装甲板四点矩形
    // for (int i = 0; i < 4; i++) {
    //     line(image2show, object.pts[i], object.pts[(i + 1) % 4], Scalar(100, 200, 0), 1);
    // }

    // 绘制目标颜色与类别
    int id = object.cls;
    int box_top_x = object.apex->x + roiRect.x;
    int box_top_y = object.apex->y + roiRect.y;
    // 绘制类别标签（全局坐标）
    Point2f box_top = global_apex[0];
    // if (object.color == 0)
    //     cv::putText(image2show, "Blue_"+to_string(id), Point(box_top_x + 2, box_top_y), cv::FONT_HERSHEY_TRIPLEX, 1,
    //                 Scalar(255, 0, 0));
    // else if (object.color == 1)
    //     cv::putText(image2show, "Red_"+to_string(id), Point(box_top_x + 2, box_top_y), cv::FONT_HERSHEY_TRIPLEX, 1,
    //                 Scalar(0, 0, 255));
    // else if (object.color == 2)
    //     cv::putText(image2show, "None_"+to_string(id), Point(box_top_x + 2, box_top_y), cv::FONT_HERSHEY_TRIPLEX, 1,
    //                 Scalar(0, 255, 0));
    if (object.color == 0)
        putText(image2show, "Blue_"+to_string(id), box_top, FONT_HERSHEY_TRIPLEX, 1, Scalar(255,0,0));
    else if (object.color == 1)
        putText(image2show, "Red_"+to_string(id), box_top, FONT_HERSHEY_TRIPLEX, 1, Scalar(0,0,255));
    else if (object.color == 2)
        putText(image2show, "None_"+to_string(id), box_top, FONT_HERSHEY_TRIPLEX, 1, Scalar(0,255,0));
}
