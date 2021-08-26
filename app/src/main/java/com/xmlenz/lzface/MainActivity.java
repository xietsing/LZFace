package com.xmlenz.lzface;


import android.Manifest;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import com.xmlenz.lzface.seeta.FaceDetector;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {

    private static final String TAG = MainActivity.class.getSimpleName();

    private CameraBridgeViewBase cameraBridgeViewBase;
    private Mat mRgba;
    private Mat mGray;

    private int cameraindex;


    public static int PERMISSION_REQ = 0x123456;

    private String[] mPermission = new String[] {
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.SYSTEM_ALERT_WINDOW
    };

    private List<String> mRequestPermission = new ArrayList<String>();


    public Mat rotate(Mat src, double angele) {
        Mat dst = new Mat(0,0, CvType.CV_8UC4);
        org.opencv.core.Point center = new  org.opencv.core.Point(src.width() / 2, src.height() / 2);
        Mat affineTrans = Imgproc.getRotationMatrix2D(center, angele, 1.0);
        Imgproc.warpAffine(src, dst, affineTrans, dst.size(), Imgproc.INTER_NEAREST);
        return dst;
    }

    private CameraBridgeViewBase.CvCameraViewListener2 mCvCameraViewListener2 = new CameraBridgeViewBase.CvCameraViewListener2() {

        private FaceDetector mFaceDetect = FaceDetector.getInstance();

        @Override
        public void onCameraViewStarted(int width, int height) {
            Log.d(TAG, "onCameraViewStarted()");
            mRgba = new Mat(height, width, CvType.CV_8UC4);
            mGray = new Mat(height, width, CvType.CV_8UC1);
            //在这里调用所有需要提前初始化的native方法
            mFaceDetect.loadEngine();
        }

        @Override
        public void onCameraViewStopped() {
            mRgba.release();
            mGray.release();

            //在这里析构和释放所有之前初始化和分配内存的native方法
            mFaceDetect.releaseEngine();
        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


            //获取到显示Mat赋值给frame
            Mat frame = inputFrame.rgba();
            //判断横竖屏用于进行图像的旋转
            if (getResources().getConfiguration().orientation == ActivityInfo.SCREEN_ORIENTATION_PORTRAIT) {
                //判断是前置摄像头还是后置摄像头,翻转的角度不一样
                switch (cameraindex) {
                    case CameraBridgeViewBase.CAMERA_ID_FRONT:
                        Core.rotate(frame, frame, Core.ROTATE_90_COUNTERCLOCKWISE);
                        Core.flip(frame,frame,1);
                        break;
                    case CameraBridgeViewBase.CAMERA_ID_BACK:
                        Core.rotate(frame, frame, Core.ROTATE_90_CLOCKWISE);
                        break;
                    default:
                        Core.rotate(frame, frame, Core.ROTATE_90_CLOCKWISE);
                        break;
                }
                //把旋转后的Mat图像根据摄像头屏幕的大小进行缩放
//                Size size = new Size(cameraView.getWidth(), cameraView.getHeight());
//                Imgproc.resize(frame, frame, size);
            }

            frame.copyTo(mRgba);
            mGray = inputFrame.gray();

            long ftStartTime = System.currentTimeMillis();

            //在这里调用处理每一张frame的native方法 记得在方法中传入的是long型的
            mFaceDetect.detect(mRgba.getNativeObjAddr());
            Log.i(TAG, "onPreviewFrame: ft costTime = " + (System.currentTimeMillis() - ftStartTime) + "ms");
            return mRgba;
        }
    };

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,  int[] grantResults) {
        // 版本兼容
        if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.M) {
            return;
        }
        if (requestCode == PERMISSION_REQ) {
            for (int i = 0; i < grantResults.length; i++) {
                for (String one : mPermission) {
                    if (permissions[i].equals(one) && grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                        mRequestPermission.remove(one);
                    }
                }
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            for (String one : mPermission) {
                if (PackageManager.PERMISSION_GRANTED != this.checkPermission(one, Process.myPid(), Process.myUid())) {
                    mRequestPermission.add(one);
                }
            }
            if (!mRequestPermission.isEmpty()) {
                this.requestPermissions(mRequestPermission.toArray(new String[mRequestPermission.size()]), PERMISSION_REQ);
            }
        }

        initView();
    }

    private void initView() {
        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(mCvCameraViewListener2);
//        cameraindex = CameraBridgeViewBase.CAMERA_ID_FRONT;
        cameraindex = CameraBridgeViewBase.CAMERA_ID_BACK;
        cameraBridgeViewBase.setCameraIndex(cameraindex);

        //在这里设置图像的大小，在手机中图像需要横屏状态，图片太大的话会卡顿
//        cameraBridgeViewBase.setMaxFrameSize(640, 480);
//        cameraBridgeViewBase.setMaxFrameSize(1920, 1080);
        cameraBridgeViewBase.setMaxFrameSize(960, 720);


        cameraBridgeViewBase.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                finish();
                return false;
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        disableCamera();
    }


    @Override
    public void onResume() {
        super.onResume();
        if (cameraBridgeViewBase != null)
            cameraBridgeViewBase.enableView();
    }

    public void onDestroy() {
        super.onDestroy();
        disableCamera();
    }

    public void disableCamera() {
        if (cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
    }
}
