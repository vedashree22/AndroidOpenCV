package com.example.stars.androidopencv;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;

import static org.opencv.features2d.Features2d.DRAW_RICH_KEYPOINTS;

public class MainActivity extends AppCompatActivity implements OnTouchListener, CvCameraViewListener2 {
    private static final int REQUEST_CODE = 100;
    TextView touch_coordinates;
    TextView touch_color;
    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;
    Mat descriptor1, descriptor2;
    MatOfKeyPoint keypoints1, keypoints2;
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba, mRef;
    private Scalar mBlobColorRgba;
    private Scalar mBlobColorHsv;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //https://stackoverflow.com/questions/38552144/how-get-permission-for-camera-in-android-specifically-marshmallow
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CODE);
        }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_tutorial_activity_surface_view);
        touch_coordinates = (TextView) findViewById(R.id.touch_coordinates);
        touch_color = (TextView) findViewById(R.id.touch_color);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        double x, y;

        int cols = mRgba.cols();
        int rows = mRgba.rows();

        double yLow = (double) mOpenCvCameraView.getHeight() * 0.2401961;
        double yHigh = (double) mOpenCvCameraView.getHeight() * 0.7696078;
        double xScale = (double) cols / (double) mOpenCvCameraView.getWidth();
        double yScale = (double) rows / (yHigh - yLow);
        x = event.getX();
        y = event.getY();
        y = y - yLow;
        x = x * xScale;
        y = y * yScale;
        if (((x < 0) || (y < 0)) || ((x > cols) || (y > rows))) return false;
        touch_coordinates.setText("X: " + Double.valueOf(x) + ", Y: " + Double.valueOf(y));

        Rect touchedRect = new Rect();

        touchedRect.x = (int) x;
        touchedRect.y = (int) y;

        touchedRect.width = 8;
        touchedRect.height = 8;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width * touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv);

        touch_color.setText("Color: %" + String.format("%02X", (int) mBlobColorRgba.val[0])
                + String.format("%02X", (int) mBlobColorRgba.val[1])
                + String.format("%02X", (int) mBlobColorRgba.val[2]));

        touch_color.setTextColor(Color.rgb((int) mBlobColorRgba.val[0],
                (int) mBlobColorRgba.val[1],
                (int) mBlobColorRgba.val[2]));

        touch_coordinates.setTextColor(Color.rgb((int) mBlobColorRgba.val[0],
                (int) mBlobColorRgba.val[1],
                (int) mBlobColorRgba.val[2]));

        return false;
    }

    private Scalar convertScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mRef = new Mat();
        descriptor1 = new Mat();
        descriptor2 = new Mat();
        keypoints1 = new MatOfKeyPoint();
        keypoints2 = new MatOfKeyPoint();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        try {
            init();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void init() throws IOException {

        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);


        AssetManager assetManager = getAssets();
        InputStream input_stream;
        input_stream = assetManager.open("image2.jpg");
        Bitmap bit = BitmapFactory.decodeStream(input_stream);
        Utils.bitmapToMat(bit, mRef);
        Imgproc.cvtColor(mRef, mRef, Imgproc.COLOR_RGB2GRAY);
        mRef.convertTo(mRef, 0);
        detector.detect(mRef, keypoints1);
        descriptor.compute(mRef, keypoints1, descriptor1);

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat matchKeypoints() {
        //Make a copy of the the input frame
        Mat inputRgb = new Mat();
        mRgba.copyTo(inputRgb);

        double min_dist = 100.0;
        // double max_dist = 0.0;
        double dist;

        //convert the image to grayscale and set the depth value to 0
        Imgproc.cvtColor(inputRgb, inputRgb, Imgproc.COLOR_RGB2GRAY);
        inputRgb.convertTo ( inputRgb, 0 );

        //detect keypoints and descriptors
        detector.detect(inputRgb, keypoints2);
        descriptor.compute(inputRgb, keypoints2, descriptor2);
        //Features2d.drawKeypoints(inputRgb, keypoints2, inputRgb, new Scalar(255, 255, 255), 3);

        //If there is no descriptors in the current frame(say a blank screen), just return the same
        // frame
        if (descriptor2.empty())
            return mRgba;

        //Match the two descriptors using BRUTEFORCE_HAMMING algorithm
        MatOfDMatch match_pairs = new MatOfDMatch();
        if (mRef.type() == inputRgb.type())
            matcher.match(descriptor1, descriptor2, match_pairs);
        else
            return mRgba;

        //distance is the score of similarity between two descriptors vectors being matched
        //we calculate the min_dist for all the match pairs and save it in min_dist
        for (int i = 0; i < match_pairs.toList().size(); i++) {
            dist = (double) match_pairs.toList().get(i).distance;
            if (dist < min_dist)
                min_dist = dist;
            // if (dist > max_dist)
            //   max_dist = dist;
        }

        //then  we create a list of good_matches whose distance would fall within our threshold
        LinkedList<DMatch> good_matches = new LinkedList();
        for (int i = 0; i < match_pairs.toList().size(); i++)
            if (match_pairs.toList ().get ( i ).distance <= (1.5 * min_dist))
                good_matches.addLast ( match_pairs.toList ().get ( i ) );

        MatOfDMatch good_match = new MatOfDMatch();
        good_match.fromList(good_matches);

        Mat outputRgb = new Mat();
        MatOfByte draw_match = new MatOfByte();

        Features2d.drawMatches(mRef, keypoints1, inputRgb, keypoints2, good_match, outputRgb,
                new Scalar(255, 0, 0), new Scalar(0, 255, 0), draw_match, DRAW_RICH_KEYPOINTS);

        Imgproc.resize(outputRgb, outputRgb, inputRgb.size());
        return outputRgb;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        return matchKeypoints();
    }


    @Override
    public void onStart() {
        super.onStart();

    }

    @Override
    public void onStop() {
        super.onStop();

    }
}

