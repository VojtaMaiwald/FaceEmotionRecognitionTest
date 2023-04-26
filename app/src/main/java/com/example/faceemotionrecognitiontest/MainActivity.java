package com.example.faceemotionrecognitiontest;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.odml.image.BitmapMlImageBuilder;
import com.google.android.odml.image.MlImage;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import android.Manifest;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Switch;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity implements CompoundButton.OnCheckedChangeListener {
    public static Boolean REGRESSION = false;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;
    private PreviewView previewView;
    private MlImage mlImage;
    private ImageView imageView;
    private Bitmap imageViewBitmap;
    private Bitmap bitmap;
    private Bitmap faceBitmap;
    private FaceDetector detector;
    private Face face;
    private Interpreter interpreter;
    private Timer timer;
    private TimerTask timerTask;
    private int width;
    private int height;
    private EmotionList<float[]> emotionList;
    private Spinner dropdown;
    Paint paintTextBlack, paintTextWhite, paintBoundBoxRed, paintBoundBoxWhite, paintCircleBlack, paintCircleWhite, paintTextBlackHalf, paintTextWhiteHalf, paintDotRed, paintDotBlue;

    private final String[] EMOTIONS = new String[]{"Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"};
    private final String[] regressors = new String[]{
            "RMSE_0.380_MnasNet_AroVal_E10_B8_A1.0_DEPTH1_Adam0.001.tflite",
            "RMSE_0.387_ShuffleNet_AroVal_E10_B8_Channels200_Adam0.001.tflite",
            "RMSE_0.390_ShuffleNetV2_AroVal_E10_B8_SC1.5_BOTTLENECK1_SGD0.01.tflite",
            "RMSE_0.392_EfficientNetB0_AroVal_E25_B8_SGD0.01.tflite",
            "RMSE_0.398_MobileNetV2_AroVal_B8_E25_D0.5_Adam_0.01.tflite",
    };
    private final String[] classifiers = new String[]{
            "PERC_56.889_MnasNet_E25_B8_A1.5_DEPTH3_Adam0.0001.tflite",
            "PERC_56.164_EfficientNetB0_E25_B8_SGD0.01.tflite",
            "PERC_55.639_DenseNet121_E25_B8_Adam0.0001.tflite",
            "PERC_55.489_EfficientNetB1_E25_B8_SGD0.01.tflite",
            "PERC_54.839_MobileNetV2_E25_B8_D_0.2.tflite",
            "PERC_54.714_GhostNet_E25_B8_Adam0.0001.tflite",
            "PERC_54.414_SqueezeNet_E25_B8_COMPR1.0_D0.2_Adam0.0001.tflite",
            "PERC_54.339_MobileNetV3Large_E25_B16_A_1.25_D_0.2.tflite",
            "PERC_53.938_MobileNetV3Small_E30_B16_A_1.25_D_0.5.tflite",
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        initApp();
        setupCamera();
        setupFaceDetector();
        setupEmotionDetector();
        setTimer();
    }

    private void killTimer() {
        if (timerTask != null) {
            timerTask.cancel();
            timerTask = null;
        }
        if (timer != null) {
            timer.cancel();
            timer.purge();
            timer = null;
        }
    }

    private void setTimer() {
        timer = new Timer();
        timerTask = new TimerTask() {
            @Override
            public void run() {
                if (imageViewBitmap == null) {
                    imageViewBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                } else if (imageCapture != null) {
                    takePicture();
                    //savePicture();
                    detectFaces();
                   //saveFaceBitmap();
                    detectEmotions();
                    drawDetections();
                }
            }
        };
        timer.scheduleAtFixedRate(timerTask, 1000, 200);
    }

    private void detectEmotions() {
        if (faceBitmap == null) {
            emotionList.removeLast();
            return;
        }
        DataType inputDataType = interpreter.getInputTensor(0).dataType();
        TensorImage tensorImage = new TensorImage(inputDataType);
        tensorImage.load(faceBitmap);
        FloatBuffer output = FloatBuffer.allocate(REGRESSION ? 2 : 8);

        try {
            long startTime = System.currentTimeMillis();
            interpreter.run(tensorImage.getBuffer(), output);
            long difference = System.currentTimeMillis() - startTime;
            Log.wtf("emotionsDetectionsTime", difference + " ms");
            emotionList.add(output.array());
        }
        catch (Exception ignored) {

        }

        Log.wtf("emotionsDetections", Arrays.toString(emotionList.getTail()));
    }

    private void drawDetections() {
        runOnUiThread(() -> {
            Bitmap tempBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            Canvas tempCanvas = new Canvas(tempBitmap);
            tempCanvas.drawBitmap(imageViewBitmap, 0, 0, null);

            float widthThird = width / 3.0f;
            float widthThreeQuarters = (width / 4.0f) * 3;
            float widthQuarter = (width / 4.0f);
            //draw emotions and probabilities
            if (REGRESSION) {
                tempCanvas.drawCircle(widthThreeQuarters - 50, widthQuarter + 50, widthQuarter, paintCircleBlack);
                tempCanvas.drawCircle(widthThreeQuarters - 50, widthQuarter + 50, widthQuarter, paintCircleWhite);
                tempCanvas.drawCircle(widthThreeQuarters - 50, widthQuarter + 50, widthQuarter / 2, paintCircleBlack);
                tempCanvas.drawCircle(widthThreeQuarters - 50, widthQuarter + 50, widthQuarter / 2, paintCircleWhite);

                tempCanvas.drawText("Surprised", widthThreeQuarters - 145, 42, paintTextBlackHalf);
                tempCanvas.drawText("Surprised", widthThreeQuarters - 145, 42, paintTextWhiteHalf);
                tempCanvas.drawText("Calm", widthThreeQuarters - 110, widthQuarter * 2 + 100, paintTextBlackHalf);
                tempCanvas.drawText("Calm", widthThreeQuarters - 110, widthQuarter * 2 + 100, paintTextWhiteHalf);
                tempCanvas.drawText("Neutral",widthThreeQuarters - 130, widthQuarter + 70, paintTextBlackHalf);
                tempCanvas.drawText("Neutral",widthThreeQuarters - 130, widthQuarter + 70, paintTextWhiteHalf);

                tempCanvas.rotate(-45, widthThreeQuarters - 50, widthQuarter + 50);
                tempCanvas.drawText("Angry", widthThreeQuarters - 145, 42, paintTextBlackHalf);
                tempCanvas.drawText("Angry", widthThreeQuarters - 145, 42, paintTextWhiteHalf);
                tempCanvas.drawText("Relaxed", widthThreeQuarters - 110, widthQuarter * 2 + 100, paintTextBlackHalf);
                tempCanvas.drawText("Relaxed", widthThreeQuarters - 110, widthQuarter * 2 + 100, paintTextWhiteHalf);
                tempCanvas.rotate(-45, widthThreeQuarters - 50, widthQuarter + 50);
                tempCanvas.drawText("Sad", widthThreeQuarters - 125, 42, paintTextBlackHalf);
                tempCanvas.drawText("Sad", widthThreeQuarters - 125, 42, paintTextWhiteHalf);
                tempCanvas.rotate(180, widthThreeQuarters - 50, widthQuarter + 50);
                tempCanvas.drawText("Happy", widthThreeQuarters - 145, 42, paintTextBlackHalf);
                tempCanvas.drawText("Happy", widthThreeQuarters - 145, 42, paintTextWhiteHalf);
                tempCanvas.rotate(-45, widthThreeQuarters - 50, widthQuarter + 50);
                tempCanvas.drawText("Excited", widthThreeQuarters - 145, 42, paintTextBlackHalf);
                tempCanvas.drawText("Excited", widthThreeQuarters - 145, 42, paintTextWhiteHalf);
                tempCanvas.drawText("Bored", widthThreeQuarters - 110, widthQuarter * 2 + 100, paintTextBlackHalf);
                tempCanvas.drawText("Bored", widthThreeQuarters - 110, widthQuarter * 2 + 100, paintTextWhiteHalf);
                tempCanvas.rotate(-45, widthThreeQuarters - 50, widthQuarter + 50);

                float[] averages = emotionList.getEmotionAverages();
                if (averages != null) {
                    averages[0] = Math.min(Math.max(averages[0], -1), 1);
                    averages[1] = Math.min(Math.max(averages[1], -1), 1);
                    tempCanvas.drawCircle(widthThreeQuarters - 50 + (widthQuarter * averages[1]), widthQuarter + 50 - (widthQuarter * averages[0]), 20, paintDotRed);
                    tempCanvas.drawCircle(widthThreeQuarters - 50 + (widthQuarter * averages[1]), widthQuarter + 50 - (widthQuarter * averages[0]), 20, paintDotBlue);
                }
            }
            else {
                float[] averages = emotionList.getEmotionAverages();
                tempCanvas.drawLine(0, 0, width, 0, paintTextBlack);
                tempCanvas.drawLine(widthThird, 0, widthThird, 640, paintTextBlack);
                tempCanvas.drawLine(widthThreeQuarters, 0, widthThreeQuarters, 640, paintTextBlack);
                tempCanvas.drawLine(0, 0, width, 0, paintTextWhite);
                for (int i = 0; i < EMOTIONS.length; i++) {
                    tempCanvas.drawText(EMOTIONS[i], 20, 60 + (i * 80), paintTextBlack);
                    tempCanvas.drawText(EMOTIONS[i], 20, 60 + (i * 80), paintTextWhite);
                    tempCanvas.drawLine(0, 80 + (i * 80), width, 80 + (i * 80), paintTextBlack);
                    tempCanvas.drawLine(0, 80 + (i * 80), width, 80 + (i * 80), paintTextWhite);
                    if (averages == null) {
                        tempCanvas.drawText("0.00 %", widthThreeQuarters + 20, 60 + (i * 80), paintTextBlack);
                        tempCanvas.drawText("0.00 %", widthThreeQuarters + 20, 60 + (i * 80), paintTextWhite);
                    }
                    else {
                        tempCanvas.drawText(String.format(Locale.US,"%.2f %%", averages[i] * 100), widthThreeQuarters + 20, 60 + (i * 80), paintTextBlack);
                        tempCanvas.drawText(String.format(Locale.US,"%.2f %%", averages[i] * 100), widthThreeQuarters + 20, 60 + (i * 80), paintTextWhite);
                        int green = Math.min(255, (int)(255 * 2 * averages[i]));
                        int red = Math.min(255, (int)(255 * 2 * (1 - averages[i])));
                        Paint paintRect = new Paint();
                        paintRect.setColor(Color.rgb(red, green, 0));
                        paintRect.setStyle(Paint.Style.FILL);
                        tempCanvas.drawRect(widthThird, i * 80, ((widthThreeQuarters - widthThird) * averages[i]) + widthThird, 80 + (i * 80), paintRect);
                    }
                }
                tempCanvas.drawLine(widthThird, 0, widthThird, 640, paintTextWhite);
                tempCanvas.drawLine(widthThreeQuarters, 0, widthThreeQuarters, 640, paintTextWhite);
            }

            //draw face bounding box
            if (face != null) {
                float widthRatio = ((float) width) / mlImage.getWidth();
                float heightRatio = ((float) height) / mlImage.getHeight();
                Rect origBounds = face.getBoundingBox();
                Rect bounds = new Rect(width - (int) (origBounds.right * widthRatio), (int) (origBounds.top * heightRatio), width - (int) (origBounds.left * widthRatio), (int) (origBounds.bottom * heightRatio));

                tempCanvas.drawRect(bounds, paintBoundBoxRed);
                Rect bounds2 = new Rect(bounds.left + 4, bounds.top + 4, bounds.right - 4, bounds.bottom - 4);
                tempCanvas.drawRect(bounds2, paintBoundBoxWhite);
            }
            imageView.setImageBitmap(tempBitmap);
        });
    }

    private void refreshFaceBitmap(Rect rect) {
        if (bitmap == null || rect.left < 0 || rect.top < 0 || rect.width() + Math.max(0, rect.left) > bitmap.getWidth() || rect.height() + Math.max(0, rect.top) > bitmap.getHeight()) {
            return;
        }
        faceBitmap = Bitmap.createScaledBitmap(Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height()), 224, 224, true);
    }

    private boolean saveFaceBitmap() {
        if (faceBitmap == null) {
            return false;
        }
        savePicture(faceBitmap);
        return true;
    }

    private void setupEmotionDetector() {
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();

        // by default it uses some kind of delegate so maybe this is not necessary to explicitly set NNAPI or GPU delegates

        //if (compatList.isDelegateSupportedOnThisDevice()) {
        //    // if the device has a supported GPU, add the GPU delegate
        //    GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        //    delegateOptions.setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
        //    //delegateOptions.setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENGL); //wait for Tensorflow Lite 2.12.0
        //    GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        //    interpreterOptions.addDelegate(gpuDelegate);
        //    Log.wtf("emotions", "GPU supported");
//
        //} else {
        //    Log.wtf("emotions", "GPU not supported");
        //}

        //if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        //    // if the device has a support in NNAPI, add the NNAPI delegate
        //    NnApiDelegate nnApiDelegate = new NnApiDelegate();
        //    interpreterOptions.setUseNNAPI(true);
        //    interpreterOptions.addDelegate(nnApiDelegate);
        //    Log.wtf("emotions", "NNAPI supported");
        //}
        //else {
        //    Log.wtf("emotions", "NNAPI not supported");
        //}

        try {
            interpreter = new Interpreter(loadModelFile(), interpreterOptions);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setupFaceDetector() {
        FaceDetectorOptions faceDetectorOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setMinFaceSize(0.15f)
                .enableTracking()
                .build();

        detector = FaceDetection.getClient(faceDetectorOptions);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd(REGRESSION ? regressors[dropdown.getSelectedItemPosition()] : classifiers[dropdown.getSelectedItemPosition()]);
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.getStartOffset(), assetFileDescriptor.getDeclaredLength());
    }

    void bindPreview() {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        imageCapture = new ImageCapture.Builder().setTargetRotation(Surface.ROTATION_0).setTargetResolution(new Size(360, 640)).build();
        cameraProvider.bindToLifecycle(this, cameraSelector, imageCapture, preview);
    }

    private void takePicture() {
        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                buffer.rewind();
                byte[] bytes = new byte[buffer.capacity()];
                buffer.get(bytes);
                byte[] clonedBytes = bytes.clone();
                bitmap = BitmapFactory.decodeByteArray(clonedBytes, 0, clonedBytes.length);
                Matrix matrix = new Matrix();
                matrix.postRotate(-90);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                mlImage = new BitmapMlImageBuilder(bitmap).build();
                image.close();
            }
        });
    }

    private void detectFaces() {
        if (mlImage == null) {
            return;
        }
        detector.process(mlImage).addOnSuccessListener(faces -> {
            if (faces.isEmpty()) {
                this.face = null;
            } else {
                this.face = faces.get(0);
                refreshFaceBitmap(this.face.getBoundingBox());
            }
        }).addOnFailureListener(e -> {
        });
    }

    private void initApp() {
        ActionBar bar = getSupportActionBar();
        if (bar != null) {
            bar.hide();
        }
        emotionList = new EmotionList<>(10, EMOTIONS.length);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        imageView = findViewById(R.id.imageView);

        paintTextBlack = new Paint();
        paintTextBlack.setColor(Color.BLACK);
        paintTextBlack.setStrokeWidth(5);
        paintTextBlack.setStyle(Paint.Style.FILL_AND_STROKE);
        paintTextBlack.setTextSize(60);

        paintTextWhite = new Paint();
        paintTextWhite.setColor(Color.WHITE);
        paintTextWhite.setStyle(Paint.Style.FILL);
        paintTextWhite.setTextSize(60);

        paintDotRed = new Paint();
        paintDotRed.setColor(Color.RED);
        paintDotRed.setStrokeWidth(5);
        paintDotRed.setStyle(Paint.Style.FILL_AND_STROKE);
        paintDotRed.setTextSize(60);

        paintDotBlue = new Paint();
        paintDotBlue.setColor(Color.BLUE);
        paintDotBlue.setStyle(Paint.Style.FILL);
        paintDotBlue.setTextSize(60);

        paintTextBlackHalf = new Paint();
        paintTextBlackHalf.setColor(Color.BLACK);
        paintTextBlackHalf.setStrokeWidth(5);
        paintTextBlackHalf.setStyle(Paint.Style.FILL_AND_STROKE);
        paintTextBlackHalf.setTextSize(50);

        paintTextWhiteHalf = new Paint();
        paintTextWhiteHalf.setColor(Color.WHITE);
        paintTextWhiteHalf.setStyle(Paint.Style.FILL);
        paintTextWhiteHalf.setTextSize(50);

        paintCircleBlack = new Paint();
        paintCircleBlack.setColor(Color.BLACK);
        paintCircleBlack.setStrokeWidth(5);
        paintCircleBlack.setStyle(Paint.Style.STROKE);
        paintCircleBlack.setTextSize(60);

        paintCircleWhite = new Paint();
        paintCircleWhite.setColor(Color.WHITE);
        paintCircleWhite.setStyle(Paint.Style.STROKE);
        paintCircleWhite.setTextSize(60);

        paintBoundBoxRed = new Paint();
        paintBoundBoxRed.setColor(Color.RED);
        paintBoundBoxRed.setStrokeWidth(5);
        paintBoundBoxRed.setStyle(Paint.Style.STROKE);

        paintBoundBoxWhite = new Paint();
        paintBoundBoxWhite.setColor(Color.WHITE);
        paintBoundBoxWhite.setStrokeWidth(3);
        paintBoundBoxWhite.setStyle(Paint.Style.STROKE);

        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.MANAGE_EXTERNAL_STORAGE}, 2);
        }

        DisplayMetrics metrics = this.getResources().getDisplayMetrics();
        width = metrics.widthPixels;
        height = metrics.heightPixels;

        ((Switch)findViewById(R.id.switchPreviewButton)).setOnCheckedChangeListener(this);
        ((Switch)findViewById(R.id.switchRegressionButton)).setOnCheckedChangeListener(this);

        dropdown = findViewById(R.id.spinnerClass);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, classifiers);
        dropdown.setAdapter(adapter);
        dropdown.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                setupEmotionDetector();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (buttonView.getId() == R.id.switchPreviewButton) {
            Preview preview = new Preview.Builder().build();
            if (isChecked) {
                previewView.setVisibility(View.INVISIBLE);
                previewView = null;
            } else {
                previewView = findViewById(R.id.previewView);
                previewView.setVisibility(View.VISIBLE);
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
            }
            CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();
            imageCapture = new ImageCapture.Builder().setTargetRotation(Surface.ROTATION_0).setTargetResolution(new Size(480, 640)).build();
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this, cameraSelector, imageCapture, preview);
        }
        if (buttonView.getId() == R.id.switchRegressionButton) {
            REGRESSION = isChecked;
            emotionList.setEmotions(REGRESSION ? 2 : 8);
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, REGRESSION ? regressors : classifiers);
            dropdown.setAdapter(adapter);
        }
    }

    private void setupCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreview();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void savePicture() {
        savePicture(this.bitmap);
    }

    private void savePicture(Bitmap bitmap) {
        if (bitmap == null) {
            return;
        }
        String path = Environment.getExternalStorageDirectory().toString();
        OutputStream fOut;
        File file = new File(path, "cosik.jpg");
        try {
            fOut = Files.newOutputStream(file.toPath());
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, fOut);
            fOut.flush();
            fOut.close();
            MediaStore.Images.Media.insertImage(getContentResolver(), file.getAbsolutePath(), file.getName(), file.getName());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onStart() {
        setTimer();
        super.onStart();
    }

    @Override
    protected void onStop() {
        killTimer();
        super.onStop();
    }

    @Override
    protected void onPause() {
        killTimer();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        killTimer();
        super.onDestroy();
    }
}