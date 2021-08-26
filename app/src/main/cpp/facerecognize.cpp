#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <seeta/FaceRecognizer.h>
#include <seeta/FaceDatabase.h>
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/Common/CStruct.h>
#include <seeta/Common/Struct.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , "Seeta", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN , "Seeta", __VA_ARGS__)

static seeta::FaceRecognizer *FR = NULL;
static seeta::FaceDetector *FD = NULL;
static seeta::FaceLandmarker *FL =NULL;
static seeta::FaceDatabase *DB =NULL;

static std::map<int64_t, std::string> GalleryIndexMap;
// recognization threshold
static float threshold = 0.6;

extern "C"
JNIEXPORT jint JNICALL
Java_com_xmlenz_lzface_seeta_FaceRecognizer_initNativeEngine(JNIEnv *env,
                                                                           jobject thiz,
                                                                           jstring detect_model_file,
                                                                           jstring marker_model_file,
                                                                           jstring recognize_model_file) {

    int b = 1;
    const char *detectModelFile = env->GetStringUTFChars(detect_model_file, NULL);
    const char *markerModelFile = env->GetStringUTFChars(marker_model_file, NULL);
    const char *recognizeModelFile = env->GetStringUTFChars(recognize_model_file, NULL);
    LOGD("===%s",detectModelFile);
    LOGD("===%s",markerModelFile);
    LOGD("===%s",recognizeModelFile);
    seeta::ModelSetting::Device device = seeta::ModelSetting::AUTO;
    int id = 0;
    seeta::ModelSetting FD_model(detectModelFile, device, id );
    seeta::ModelSetting PD_model(markerModelFile, device, id );
    seeta::ModelSetting FR_model(recognizeModelFile, device, id );

    FR = new seeta::FaceRecognizer(FR_model);
    FD = new seeta::FaceDetector(FD_model);
    FL = new seeta::FaceLandmarker(PD_model);
    DB = new seeta::FaceDatabase(FR_model,2,16);

    FR->set(seeta::FaceRecognizer::PROPERTY_ARM_CPU_MODE, 1);
    //set face detect threshold
    FR->set(seeta::FaceRecognizer::PROPERTY_NUMBER_THREADS, 0.60f);

    env->ReleaseStringUTFChars(detect_model_file, detectModelFile);
    env->ReleaseStringUTFChars(marker_model_file, markerModelFile);
    env->ReleaseStringUTFChars(recognize_model_file, recognizeModelFile);
    int res = EXIT_SUCCESS;

    return (jint)res;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_xmlenz_lzface_seeta_FaceRecognizer_nativeRegisterFace(JNIEnv *env, jobject instance, jobject faceList) {

    if(NULL == FR) {
        LOGW("FR is NULL");
        return;
    }

    jclass jArrayList = env->GetObjectClass(faceList);
    jmethodID jArrayList_get = env->GetMethodID(jArrayList, "get", "(I)Ljava/lang/Object;");
    jmethodID jArrayList_size = env->GetMethodID(jArrayList, "size", "()I");
    jint len = env->CallIntMethod(faceList, jArrayList_size);
    LOGD("face len: %d", len);
//    GalleryIndexMap.clear();
    for (int i = 0; i < len; i++) {
        jstring filepath_ = (jstring) env->CallObjectMethod(faceList, jArrayList_get, i);
        const char *filepath = env->GetStringUTFChars(filepath_, 0);
        LOGD("filepath: %s", filepath);

        cv::Mat image = cv::imread( filepath );

        SeetaImageData simage;
        simage.height = image.rows;
        simage.width = image.cols;
        simage.channels = image.channels();
        simage.data = image.data;

        auto faces = FD->detect(simage);
        LOGD("faces size: %d", faces.size);
        char name[100];
        sprintf(name,"%d:",i+1);
        if(faces.size>0)
        {
            int i =0;
            auto &face = faces.data[i];
            auto points = FL->mark(simage, face.pos);

            if(points.size()>0)
            {
                for (auto &point : points)
                {
                    auto id = DB->Register( simage,&point );
                    LOGD("Registered id = %ld", id);
                    if(id >= 0) {
                        GalleryIndexMap.insert( std::make_pair( id, name ) );
                        break;
                    }
                }
            }
        }


        env->ReleaseStringUTFChars(filepath_, filepath);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_xmlenz_lzface_seeta_FaceRecognizer_nativeRecognition(JNIEnv *env, jobject instance, jlong addr) {

    if(NULL == FR) {
        LOGW("FR is NULL");
        return;
    }
    // TODO
    cv::Mat &frame = *(cv::Mat *) addr;
    cv::Mat rgb_img;
    cv::cvtColor(frame, rgb_img, cv::COLOR_RGBA2BGR);

//    seeta::ImageData image = rgb_img;

    SeetaImageData image;
    image.height = rgb_img.rows;
    image.width = rgb_img.cols;
    image.channels = rgb_img.channels();
    image.data = rgb_img.data;

//    std::shared_ptr<float> extract(
//            seeta::FaceRecognizer *fr,
//            const SeetaImageData &image,
//            const std::vector<SeetaPointF> &points) {
//                std::shared_ptr<float> features(
//                    new float[fr->GetExtractFeatureSize()],
//                    std::default_delete<float[]>());
//                fr->Extract(image, points.data(), features.get());
//                return features;
//            }

    // Detect all faces
    auto faces = FD->detect( image );

    for (int i = 0; i < faces.size; ++i)
    {
        auto &face = faces.data[i];
        // Query top 1
        int64_t index = -1;
        float similarity = 0;

        auto points = FL->mark (image, face.pos);


        auto queried = DB->QueryTop( image, points.data(), 1, &index, &similarity );

//        cv::rectangle( frame, cv::Rect( face.pos.x, face.pos.y, face.pos.width, face.pos.height ), CV_RGB( 0, 0, 255 ), 1 );
//        for (int i = 0; i < 5; ++i)
//        {
//            auto &point = points[i];
//            cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
//        }

        // no face queried from database
        if (queried < 1) continue;

        // similarity greater than threshold, means recognized
        LOGW("similarity: %f", similarity);
        if( similarity > threshold )
        {
            std::string name = GalleryIndexMap[index];
            LOGD("name: %s", name.c_str());


            std::stringstream name_1;
            name_1 << name<< " like=" << similarity;
            cv::rectangle( frame, cv::Rect( face.pos.x, face.pos.y, face.pos.width, face.pos.height ), CV_RGB( 0, 0, 255 ), 1 );
            for (int i = 0; i < 5; ++i)
            {
                auto &point = points[i];
                cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
            }
            cv::putText( frame, name_1.str(), cv::Point( face.pos.x, face.pos.y - 5 ), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB( 0, 0, 0 ) );
        }
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_xmlenz_lzface_seeta_FaceRecognizer_releaseNativeEngine(JNIEnv *env, jobject instance) {
    if(NULL != FR) {
        delete FR;
    }
    delete FD;
    delete FL;
    int ret = EXIT_SUCCESS;
    return (jint)ret;
}