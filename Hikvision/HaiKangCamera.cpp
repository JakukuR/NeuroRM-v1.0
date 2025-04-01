#include "./include/HaiKangCamera.h"
#define FMT_HEADER_ONLY
#include <fmt/format.h>
HaiKangCamera::HaiKangCamera()
{
}

HaiKangCamera::~HaiKangCamera()
{
    nRet = MV_CC_FreeImageBuffer(handle, (&pFrame));
    if(nRet == MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::blue), "[Camera] free image buffer succeed!\n");
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] free image buffer failed!\n");
    }
} 

int HaiKangCamera::StartDevice(int serial_number)
{   //打开设备
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    //枚举设备
    nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if(MV_OK != nRet)
    {
        fmt::print(fmt::fg(fmt::color::red), "EnumDevices failed! nRet [%x]\n", nRet);
        return -1;
    }

    if(stDeviceList.nDeviceNum > 0) //设备数量不为0
    {
        // for(int i = 0; i < stDeviceList.nDeviceNum; i++)
        // {
        //     printf("[device %d]:\n", i);
        //     MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
        //     if(NULL == pDeviceInfo)
        //     {
        //         printf("pDeviceInfo is NULL!\n");
        //         return -1;
        //     }
        //     HaiKangCamera::PrintDeviceInfo(pDeviceInfo);
        // }

        //打开设备
        //选择设备并创建句柄
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[serial_number]);
        if(MV_OK != nRet)
        {
            fmt::print(fmt::fg(fmt::color::red), "CreateHandle failed! nRet [%x]\n", nRet);
            return -1;
        }

        //打开设备
        nRet = MV_CC_OpenDevice(handle);
        if(MV_OK == nRet)
        {
            fmt::print(fmt::fg(fmt::color::blue), "[Camera] open succeed!\n");
            return serial_number;
        }
        if(MV_OK != nRet)
        {
            fmt::print(fmt::fg(fmt::color::red), "[Camera] open failed!\n");
            return -1;
        }
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] Find No Devices!\n");
        return -1;
    }

    return serial_number;
}

bool HaiKangCamera::SetStreamOn()
{   //开始采集
    //设置触发模式为off
    this->nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if(MV_OK != nRet)
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] SetEnumValue failed! nRet [%x]!\n", nRet);
        return false;
    }

    //设置采集模式为连续采集
    this->nRet = MV_CC_SetEnumValue(handle, "AcquisitionMode", 2);
    if(MV_OK != nRet)
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] SetEnumValue failed! nRet [%x]!\n", nRet);
        return false;
    }

    // //获取数据包大小
    // MVCC_INTVALUE stParam;
    // memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    // nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    // if (MV_OK != nRet)
    // {
    //     printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
    //     return false;
    // }
    // g_nPayloadSize = stParam.nCurValue;

    //开始取流
    nRet = MV_CC_StartGrabbing(handle);
    if(MV_OK == nRet)
    {
        fmt::print(fmt::fg(fmt::color::blue), "[Camera] StartGrabbing succeed!\n");
        return true;
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] StartGrabbing failed! nRet [%x]\n", nRet);
        return false;
    }
    
    //创建相机图像采集线程
    // pthread_t tid;
    // nRet = pthread_create(&tid, NULL, HaiKangCamera::WorkThread, handle);
    // if(MV_OK == nRet)
    // {
    //     printf("pthread_create succeed!\n");
    //     return true;
    // }
    // else
    // {
    //     printf("pthread_create failed! nRet [%d]\n", nRet);
    //     return false;
    // }
}

bool HaiKangCamera::SetResolution(int width, int height)
{   
    nRet = MV_OK;

    //设置像素格式为 RGB8
    nRet = MV_CC_SetPixelFormat(handle, PixelType_Gvsp_RGB8_Packed);
    if(nRet != MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] setPixelFormat failed! nRet [%x]\n", nRet);
        return false;
    }

    // 设置图像宽度
    nRet = MV_CC_SetIntValue(this->handle, "Width", width);
    if(nRet != MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] setResolution failed! nRet [%x]\n", nRet);
        return false;
    }

    // 设置图像高度
    nRet = MV_CC_SetIntValue(this->handle, "Height", height);
    if(nRet == MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::blue), "[Camera] setResolution succeed! nRet [%x]\n", nRet);
        return true;
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[Camera] setResolution failed! nRet [%x]\n", nRet);
        return false;
    }
}

bool HaiKangCamera::SetExposureTime(float ExposureTime)
{   //设置曝光时间
    nRet = MV_CC_SetFloatValue(this->handle, "ExposureTime", ExposureTime);
    if(nRet == MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::green), "[CAMERA] 曝光值设置成功\n");
        return true;
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 曝光值设置失败\n");
        return false;
    }
}
bool HaiKangCamera::SetFPS(float fps){
    nRet = MV_CC_SetFrameRate(this->handle, fps);
    if(nRet == MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::green), "[CAMERA] fps set success\n");
        return true;
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] fps set failed\n");
        return false;
    }
}
bool HaiKangCamera::GetImageParam(CameraAndImageParam &camera_and_image_param) {
    bool state_word = true;
    if (MV_CC_GetWidth(this->handle, &camera_and_image_param.width_message_) != MV_OK) {
        perror("MV_CC_GetWidth failed!\n");
        state_word = false;
    }else
        printf("MV_CC_GetWidth: %d \t", camera_and_image_param.width_message_.nCurValue);

    if (MV_CC_GetHeight(this->handle, &camera_and_image_param.height_message_) != MV_OK) {
        printf("MV_CC_GetHeight failed!\n");
        state_word = false;
    }else
        printf("MV_CC_GetHeight: %d \t", camera_and_image_param.height_message_.nCurValue);

    if (MV_CC_GetPixelFormat(this->handle, &camera_and_image_param.pixel_format_message_) != MV_OK) {
        perror("MV_CC_GetPixelFormat failed!\n");
        state_word = false;
    }else
        printf("MV_CC_GetPixelFormat: %d \t", camera_and_image_param.pixel_format_message_.nCurValue);

    if (MV_CC_GetFrameRate(this->handle, &camera_and_image_param.frame_rate_message_) != MV_OK) {
        perror("MV_CC_GetFrameRate failed!\n");
        state_word = false;
    }else
        printf("MV_CC_GetFrameRate: %f \t", camera_and_image_param.frame_rate_message_.fCurValue);

    if (MV_CC_GetExposureTime(this->handle, &camera_and_image_param.exposure_time_) != MV_OK) {
        perror("MV_CC_GetExposureTime failed!\n");
        state_word = false;
    }else
        printf("MV_CC_GetExposureTime: %f \t", camera_and_image_param.exposure_time_.fCurValue);


    printf("\n");

    return state_word;
}
bool HaiKangCamera::SetGAIN(int value, int ExpGain)
{   //曝光增益
    if(value == 0)
    {
        nRet = MV_CC_SetEnumValue(handle, "GainAuto", R_CHANNEL);
        cout<<nRet<<endl;
    }
    else if(value == 1)
    {
        nRet = MV_CC_SetEnumValue(handle, "GainMode", G_CHANNEL);
    }
    else if(value == 2)
    {
        nRet = MV_CC_SetEnumValue(handle, "GainMode", B_CHANNEL);
    }
    else
    {
        nRet = MV_CC_SetFloatValue(handle, "Gain", ExpGain);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::blue), "[CAMERA] 设置曝光增益成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置曝光增益失败22！\n");
            return false;
        }
    }

    nRet = MV_CC_SetFloatValue(handle, "Gain", ExpGain);
    if(nRet == MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::blue), "[CAMERA] 设置曝光增益成功！\n");
        return true;
    }
    else
    {
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置曝光增益失败11！\n");
        cout<<nRet;
        return false;
    }
}

bool HaiKangCamera::Set_Auto_BALANCE()
{   //自动白平衡（具有记忆功能）
    this->nRet = MV_CC_SetEnumValue(this->handle, "BalanceWhiteAuto", 1);
    if(nRet != MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::blue), "[CAMERA] 自动白平衡设置失败！\n");
        return false;
    }
    else
    {   
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 自动白平衡设置成功！\n");
        return true;
    }
}

bool HaiKangCamera::Set_BALANCE(int value, unsigned int value_number)
{   //手动白平衡（具有记忆功能））
    //关闭自动白平衡
    this->nRet = MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF);
    if(nRet != MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭自动白平衡失败！\n");
        return false;
    }

    //设置RGB三通道白平衡值
    if(value == 0)
    {
        this->nRet = MV_CC_SetBalanceRatioRed(handle, value_number);
        
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::blue), "[CAMERA] set R_Balance succeed！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] set R_Balance failed！\n");
            return false;
        }
    }
    else if(value == 1)
    {
        this->nRet = MV_CC_SetBalanceRatioGreen(handle, value_number);

        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::blue), "[CAMERA] set G_Balance succeed！\n");
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] set G_Balance failed！\n");
            return false;
        }
    }
    else if(value == 2)
    {
        this->nRet = MV_CC_SetBalanceRatioBlue(handle, value_number);
        
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::blue), "[CAMERA] set B_Balance succeed！\n");
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] set B_Balance failed！\n");
            return false;
        }
    }
    return true;
}

bool HaiKangCamera::Set_Gamma(bool set_status,double dGammaParam)
{   //设置Gamma值
    if(set_status == true)
    {
        nRet = MV_CC_SetEnumValue(handle, "Gamma", 1);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置Gamma值成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置Gamma值失败！\n");
            return false;
        }
    }
    else
    {
        nRet = MV_CC_SetEnumValue(handle, "Gamma", 0);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭Gamma值成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭Gamma值失败！\n");
            return false;
        }
    }
}

bool HaiKangCamera::Color_Correct(bool value)
{   //设置色彩校正
    if(value == true)
    {
        nRet = MV_CC_SetEnumValue(handle, "ColorCorrection", 1);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置色彩校正成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA]设置色彩校正失败！\n");
            return false;
        }
    }
    else
    {
        nRet = MV_CC_SetEnumValue(handle, "ColorCorrection", 0);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭色彩校正成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭色彩校正失败！\n");
            return false;
        }
    }
}

bool HaiKangCamera::Set_Contrast(bool set_status,int dContrastParam)
{   //设置对比度
    if(set_status == true)
    {
        nRet = MV_CC_SetEnumValue(handle, "Contrast", 1);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置对比度成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 设置对比度失败！\n");
            return false;
        }
    }
    else
    {
        nRet = MV_CC_SetEnumValue(handle, "Contrast", 0);
        if(nRet == MV_OK)
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭对比度成功！\n");
            return true;
        }
        else
        {
            fmt::print(fmt::fg(fmt::color::red), "[CAMERA] 关闭对比度失败！\n");
            return false;
        }
    }
}

bool HaiKangCamera::UpdateTimestampOffset(std::chrono::_V2::steady_clock::time_point time_start)
{   //计算时间戳偏移
    std::chrono::_V2::steady_clock::time_point time_end = std::chrono::_V2::steady_clock::now();
    std::chrono::duration<double> time_span = time_end - time_start;
    timestamp_offset = time_span.count() * 1000;
    return true;
}

int HaiKangCamera::Get_TIMESTAMP()
{   //获取时间戳
    std::chrono::_V2::steady_clock::time_point time_start = std::chrono::_V2::steady_clock::now();
    return ((int)time_start.time_since_epoch().count() - timestamp_offset);
}

static constexpr int MAX_IMAGE_DATA_SIZE = (4 * 1280 * 1024); //四通道 长 宽

bool HaiKangCamera::GetMat(cv::Mat &Src)
{
    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    unsigned int nDataSize = MAX_IMAGE_DATA_SIZE;
    unsigned char *pData = new unsigned char[MAX_IMAGE_DATA_SIZE];

    // 从相机获取一帧图像
    nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, &stImageInfo, 1000);
    if (nRet != MV_OK)
    {
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] Failed to get frame [%x]\n", nRet);
        delete[] pData;
        return false;
    }

    try {
        cv::Mat temp;
        // 根据像素格式创建 Mat
        if (stImageInfo.enPixelType == PixelType_Gvsp_RGB8_Packed)
        {
            temp = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData).clone();
            // 将RGB转换为BGR
            cv::cvtColor(temp, Src, cv::COLOR_RGB2BGR);
        }
        else if (stImageInfo.enPixelType == PixelType_Gvsp_Mono8)
        {
            Src = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData).clone();
        }
        else 
        {
            // 如果是其他格式，需要先转换为 RGB8
            MV_CC_PIXEL_CONVERT_PARAM stConvertParam = {0};
            unsigned char* pDataForRGB = new unsigned char[stImageInfo.nWidth * stImageInfo.nHeight * 3];
            
            stConvertParam.nWidth = stImageInfo.nWidth;
            stConvertParam.nHeight = stImageInfo.nHeight;
            stConvertParam.pSrcData = pData;
            stConvertParam.nSrcDataLen = stImageInfo.nFrameLen;
            stConvertParam.enSrcPixelType = stImageInfo.enPixelType;
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
            stConvertParam.pDstBuffer = pDataForRGB;
            stConvertParam.nDstBufferSize = stImageInfo.nWidth * stImageInfo.nHeight * 3;

            nRet = MV_CC_ConvertPixelType(handle, &stConvertParam);
            if (nRet == MV_OK)
            {
                temp = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pDataForRGB).clone();
                // 将RGB转换为BGR
                cv::cvtColor(temp, Src, cv::COLOR_RGB2BGR);
            }
            
            delete[] pDataForRGB;
        }
    }
    catch (cv::Exception& e) {
        fmt::print(fmt::fg(fmt::color::red), "[CAMERA] OpenCV error: {}\n", e.what());
        delete[] pData;
        return false;
    }

    delete[] pData;
    return true;
}