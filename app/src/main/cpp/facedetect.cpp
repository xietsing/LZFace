#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/MaskDetector.h>
#include <seeta/Common/CStruct.h>
#include <seeta/Common/Struct.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>
#include <opencv2/core.hpp>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , "Seeta", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN , "Seeta", __VA_ARGS__)

static seeta::FaceDetector *FD;
static seeta::FaceLandmarker *FL;
static seeta::MaskDetector *MD;


SeetaFaceInfo getMaxFaceRect(SeetaFaceInfoArray facelist)
{
    int len  = facelist.size;
//    int maxnow = 0;
    for (int i = 0; i < len-1; i++)
    {
        for(int j =0; j< len-1-i; j++)
        {
            if( facelist.data[j].pos.width > facelist.data[j+1].pos.width)
            {
                auto &tempface =  facelist.data[j+1];
                facelist.data[j+1] =  facelist.data[j];
                facelist.data[j] = tempface;

            }
        }
    }
    return facelist.data[len-1];
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_xmlenz_lzface_seeta_FaceDetector_initFaceDetection(JNIEnv *env, jobject instance,
                                                              jstring detectModelFile_, jstring markerModelFile_) {
    const char *detectModelFile = env->GetStringUTFChars(detectModelFile_, 0);
    const char *markerModelFile = env->GetStringUTFChars(markerModelFile_, 0);
    seeta::ModelSetting::Device device = seeta::ModelSetting::AUTO;


    int id = 0;
    seeta::ModelSetting FD_model( detectModelFile, device, id );
    seeta::ModelSetting FL_model( markerModelFile, device, id );

    FD = new seeta::FaceDetector(FD_model);
    FL = new seeta::FaceLandmarker(FL_model);

    FD->set(seeta::FaceDetector::PROPERTY_ARM_CPU_MODE, 1);
    FD->set(seeta::FaceDetector::PROPERTY_THRESHOLD, 0.65f);//0.65

    int res = EXIT_SUCCESS;
    env->ReleaseStringUTFChars(detectModelFile_, detectModelFile);
    env->ReleaseStringUTFChars(markerModelFile_, markerModelFile);
    return (jint)res;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_xmlenz_lzface_seeta_FaceDetector_applyFaceDetection(JNIEnv *env, jobject instance, jlong addr) {

    // TODO
    cv::Mat &img = *(cv::Mat *) addr;
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_RGBA2BGR);
//    cv::Mat rgb_img;
//    cv::transpose(rgb_img0,rgb_img); //roate 90...

    SeetaImageData simage;
    simage.height = rgb_img.rows;
    simage.width = rgb_img.cols;
    simage.channels = rgb_img.channels();
    simage.data = rgb_img.data;

    if(NULL == FD) {
        LOGW("FD is NULL");
        return;
    }

    auto faces = FD->detect(simage);
    LOGD("faces size: %d", faces.size);
    if(faces.size<=0)
        return;
    auto face = getMaxFaceRect(faces);

//    for (int i = 0; i < faces.size; ++i)
    {
//        auto &face = faces.data[i];
        auto points = FL->mark(simage, face.pos);
//        cv::rectangle(img, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(0, 0, 255), 2);
        // Draw Face Rect
        cv::Point top;
        top.x = face.pos.x;
        top.y = face.pos.y;
        int w = face.pos.width;
        int h = face.pos.height;
        cv::line(img, top, cv::Point(top.x+w/4,top.y),cv::Scalar(255, 255, 255), 3);
        cv::line(img, top, cv::Point(top.x ,top.y + h/4),cv::Scalar(255, 255, 255), 3);
        cv::line(img, cv::Point(top.x,top.y +h), cv::Point(top.x+w/4,top.y +h),cv::Scalar(255, 255, 255), 3);
        cv::line(img, cv::Point(top.x,top.y +h), cv::Point(top.x ,top.y + h/4*3),cv::Scalar(255, 255, 255), 3);
        cv::line(img, cv::Point(top.x + w,top.y), cv::Point(top.x+w*3/4,top.y),cv::Scalar(255, 255, 255), 3);
        cv::line(img, cv::Point(top.x + w,top.y), cv::Point(top.x+w ,top.y + h/4),cv::Scalar(255, 255, 255), 3);
        cv::line(img, cv::Point(top.x + w,top.y +h), cv::Point(top.x+w*3/4,top.y +h),cv::Scalar(255, 255, 255), 3);
        cv::line(img, cv::Point(top.x + w,top.y +h), cv::Point(top.x +w,top.y + h/4*3),cv::Scalar(255, 255, 255), 3);



        //不画出特征点阵列
//        for (auto &point : points)
//        {
//            cv::circle(img, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
//        }
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_xmlenz_lzface_seeta_FaceDetector_releaseFaceDetection(JNIEnv *env, jobject instance) {
    delete FD;
    delete FL;
    int ret = EXIT_SUCCESS;
    return (jint)ret;
}