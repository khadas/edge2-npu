package com.wesion.demo;

import static java.lang.Thread.sleep;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.ImageView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MainActivity extends Activity {

    private static final String TAG = "IdentifyActivity";
    private SurfaceView mSurfaceview;
    private SurfaceHolder mSurfaceholder;
    private static final int mPreviewWidth = 1920;
    private static final int mPreviewHeight = 1080;
    private int mVideoWidth;
    private int mVideoHeight;
    private ExecutorService executorService = Executors.newSingleThreadExecutor();
    private Future mFuture;
    private Camera mCamera = null;
    public int mFlip = 1;
    private ImageView mDetectResultView;
    private Bitmap mDetectResultBitmap = null;
    private Canvas mDetectResultCanvas = null;
    private Paint mDetectResultPaint = null;
    private Paint mDetectResultTextPaint = null;
    private PorterDuffXfermode mPorterDuffXfermodeClear;
    private PorterDuffXfermode mPorterDuffXfermodeSRC;
    public static final int OBJ_NUMB_MAX_SIZE = 128;
    private String mFileDirPath;
    private String mLpdModel = "lpd.rknn";
    private String mLpdLabel = "lpd_class.txt";
    private String mLprModel = "lpr.rknn";
    private String mLprLabel = "lpr_class.txt";
    private String mLpcModel = "lpc.rknn";
    private String mLpcLabel = "lpc_class.txt";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mSurfaceview = (SurfaceView) findViewById(R.id.surfaceViewCamera);
        mDetectResultView = (ImageView) findViewById(R.id.canvasView);

        mFileDirPath = getCacheDir().getAbsolutePath();

        createFile(mLpdModel, R.raw.lpd);
        createFile(mLprModel, R.raw.lpr);
        createFile(mLpcModel, R.raw.lpc);
        createFile(mLpdLabel, R.raw.lpd_class);
        createFile(mLprLabel, R.raw.lpr_class);
        createFile(mLpcLabel, R.raw.lpc_class);

        Recognition.native_init(mFileDirPath + "/" + mLpdModel, mFileDirPath + "/" + mLpdLabel,
                 mFileDirPath + "/" + mLprModel, mFileDirPath + "/" + mLprLabel,
                 mFileDirPath + "/" + mLpcModel, mFileDirPath + "/" + mLpcLabel);
    }

    private void createFile(String fileName, int id) {
        String filePath = mFileDirPath + "/" + fileName;
        try {
            File dir = new File(mFileDirPath);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            File file = new File(filePath);
            if (!file.exists() || isFirstRun()) {
                InputStream ins = getResources().openRawResource(id);// 通过raw得到数据资源
                FileOutputStream fos = new FileOutputStream(file);
                byte[] buffer = new byte[8192];
                int count = 0;
                while ((count = ins.read(buffer)) > 0) {
                    fos.write(buffer, 0, count);
                }
                fos.close();
                ins.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private boolean isFirstRun() {
        SharedPreferences sharedPreferences = getSharedPreferences("setting", MODE_PRIVATE);
        boolean isFirstRun = sharedPreferences.getBoolean("isFirstRun", true);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        if (isFirstRun) {
            editor.putBoolean("isFirstRun", false);
            editor.commit();
        }
        return isFirstRun;
    }

    @Override
    public void onStart() {
        super.onStart();
        openCamera();
    }

    @Override
    public void onPause() {
        super.onPause();
        releaseCamera();
    }

    private Camera.Size getOptimalPreviewSize(List<Camera.Size> sizes, int w, int h) {
        final double aspectTolerance = 0.1;
        double targetRatio = (double) w / h;
        if (sizes == null) {
            return null;
        }
        Camera.Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;
        int targetHeight = h;
        // Try to find an size match aspect ratio and size
        for (Camera.Size size : sizes) {
            double ratio = (double) size.width / size.height;
            if (Math.abs(ratio - targetRatio) > aspectTolerance) {
                continue;
            }
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }
        // Cannot find the one match the aspect ratio, ignore the requirement
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Camera.Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }
        return optimalSize;
    }

    private void openCamera() {
        if (this.checkCameraHardware(this)) {
            try {
                mCamera = Camera.open(0);
                mSurfaceholder = mSurfaceview.getHolder();
                mSurfaceholder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
                mSurfaceholder.addCallback(new surfaceholderCallbackBack());

                if (mCamera != null && mSurfaceholder != null) {
                    Camera.Parameters params = mCamera.getParameters();
                    List<Camera.Size> sizeList = params.getSupportedPreviewSizes();
                    final Camera.Size optionSize = getOptimalPreviewSize(sizeList, mPreviewWidth, mPreviewHeight);
                    if (optionSize.width == mPreviewWidth && optionSize.height == mPreviewHeight) {
                        mVideoWidth = mPreviewWidth;
                        mVideoHeight = mPreviewHeight;
                    } else {
                        mVideoWidth = optionSize.width;
                        mVideoHeight = optionSize.height;
                    }
                    params.setPreviewSize(mVideoWidth, mVideoHeight);
                    mCamera.setParameters(params);
                    mCamera.setPreviewCallback(new Camera.PreviewCallback() {
                        @Override
                        public void onPreviewFrame(byte[] data, Camera cam) {
                            onGetCameraData(data, cam, mVideoWidth, mVideoHeight);
                        }
                    });
                    mCamera.setPreviewDisplay(mSurfaceholder);
                    mCamera.setDisplayOrientation(0);
                    mCamera.startPreview();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static int sp2px(float spValue) {
        Resources r = Resources.getSystem();
        final float scale = r.getDisplayMetrics().scaledDensity;
        return (int) (spValue * scale + 0.5f);
    }

    private void showTrackSelectResults(int width, int height, DetectResultGroup detectResultGroup) {

        if (mDetectResultBitmap == null) {
            mDetectResultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            mDetectResultCanvas = new Canvas(mDetectResultBitmap);

            mDetectResultPaint = new Paint();
            mDetectResultPaint.setColor(0xff416FDA);
            mDetectResultPaint.setStrokeJoin(Paint.Join.ROUND);
            mDetectResultPaint.setStrokeCap(Paint.Cap.ROUND);
            mDetectResultPaint.setStrokeWidth(4);
            mDetectResultPaint.setStyle(Paint.Style.STROKE);
            mDetectResultPaint.setTextAlign(Paint.Align.LEFT);
            mDetectResultPaint.setTextSize(sp2px(10));
            mDetectResultPaint.setTypeface(Typeface.SANS_SERIF);
            mDetectResultPaint.setFakeBoldText(false);

            mDetectResultTextPaint = new Paint();
            mDetectResultTextPaint.setColor(Color.RED);
            mDetectResultTextPaint.setStrokeWidth(2);
            mDetectResultTextPaint.setTextAlign(Paint.Align.LEFT);
            mDetectResultTextPaint.setTextSize(sp2px(12));
            mDetectResultTextPaint.setTypeface(Typeface.SANS_SERIF);
            mDetectResultTextPaint.setFakeBoldText(false);
            mPorterDuffXfermodeClear = new PorterDuffXfermode(PorterDuff.Mode.CLEAR);
            mPorterDuffXfermodeSRC = new PorterDuffXfermode(PorterDuff.Mode.SRC);
        }

        // clear canvas
        mDetectResultPaint.setXfermode(mPorterDuffXfermodeClear);
        mDetectResultCanvas.drawPaint(mDetectResultPaint);
        mDetectResultPaint.setXfermode(mPorterDuffXfermodeSRC);

        for (int i = 0; i < detectResultGroup.count; ++i) {
            RectF detection = new RectF();
            detection.left = detectResultGroup.boxes[i * 4 + 0];
            detection.top = detectResultGroup.boxes[i * 4 + 1];
            detection.right = detectResultGroup.boxes[i * 4 + 2];
            detection.bottom = detectResultGroup.boxes[i * 4 + 3];

            byte[] licensePlate = new byte[detectResultGroup.lpLen[i]];
            for (int j = 0; j < detectResultGroup.lpLen[i]; j++) {
                licensePlate[j] = detectResultGroup.lpInfo[i * 36 + j];
            }

            String licensePlateStr = new String(licensePlate);
            mDetectResultCanvas.drawRect(detection, mDetectResultPaint);
            Log.e(TAG, "licensePlateStr " + licensePlateStr);
            mDetectResultCanvas.drawText(licensePlateStr, detection.left+5, detection.bottom-5, mDetectResultTextPaint);
        }

        mDetectResultView.setScaleType(ImageView.ScaleType.FIT_XY);
        mDetectResultView.setImageBitmap(mDetectResultBitmap);
    }

    private void onGetCameraData(final byte[] data, final Camera camera, final int width, final int height) {
        if (mFuture != null && !mFuture.isDone()) {
            return;
        }

        mFuture = executorService.submit(new Runnable() {
            @Override
            public void run() {
                Camera.Parameters parameters = camera.getParameters();
                int format = parameters.getPreviewFormat();
                final YuvImage image = new YuvImage(data, format, width, height, null);
                ByteArrayOutputStream os = new ByteArrayOutputStream(data.length);
                if (!image.compressToJpeg(new Rect(0, 0, width, height), 100, os)) {
                    return;
                }

                byte[] imageBytes = os.toByteArray();
                if (imageBytes != null) {
                    DetectResultGroup detectResultGroup = new DetectResultGroup();
                    detectResultGroup.count = 0;
                    detectResultGroup.scores = new float[OBJ_NUMB_MAX_SIZE];
                    detectResultGroup.lpLen = new int[OBJ_NUMB_MAX_SIZE];
                    detectResultGroup.boxes = new int[OBJ_NUMB_MAX_SIZE * 4];
                    detectResultGroup.lpInfo = new byte[OBJ_NUMB_MAX_SIZE * 36];
                    detectResultGroup.count = Recognition.native_identify(width, height, 3, mFlip, imageBytes, detectResultGroup.lpLen, detectResultGroup.scores, detectResultGroup.boxes, detectResultGroup.lpInfo);
                    showTrackSelectResults(width, height, detectResultGroup);
                }

            }
        });
    }


    private boolean checkCameraHardware(Context context) {
        if (context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)) {
            // this device has a camera
            return true;
        } else {
            // no camera on this device
            return false;
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        releaseCamera();
        Recognition.native_deInit();
    }

    private void releaseCamera() {
        if (mCamera != null) {
            mCamera.setPreviewCallback(null);
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
            mSurfaceholder = null;
        }
    }

    class surfaceholderCallbackBack implements SurfaceHolder.Callback {
        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            int cameraCount = Camera.getNumberOfCameras();
            if (cameraCount > 0 && mCamera != null) {
                try {
                    mCamera.setPreviewDisplay(holder);
                    mCamera.startPreview();
                } catch (IOException e) {
                    e.printStackTrace();
                    mCamera.release();
                }
            }
        }

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            if(null != mCamera) {
                mCamera.setPreviewCallback(null);
                mCamera.stopPreview();
                mCamera.release();
                mCamera = null;
            }
        }
    }

}
