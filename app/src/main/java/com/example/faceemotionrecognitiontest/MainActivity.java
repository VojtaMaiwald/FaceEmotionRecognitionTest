package com.example.faceemotionrecognitiontest;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.Task;
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
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.widget.ImageView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
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
    private float[] emotionsProbabilities;

    private final String[] EMOTIONS = new String[]{"Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        initApp();

        setupCamera();

        setupFaceDetector();

        setupEmotionDetector();

        setTimer();
    }

    private void detectEmotions() {
        if (faceBitmap == null) {
            return;
        }
        DataType inputDataType = interpreter.getInputTensor(0).dataType();
        TensorImage tensorImage = new TensorImage(inputDataType);
        tensorImage.load(faceBitmap);
        FloatBuffer output = FloatBuffer.allocate(8);
        interpreter.run(tensorImage.getBuffer(), output);
        emotionsProbabilities = output.array();
        Log.wtf("emotions", Arrays.toString(emotionsProbabilities));
    }

    private void killTimer() {
        timerTask.cancel();
        timer.cancel();
        timer.purge();
        timer = null;
        timerTask = null;
    }

    private void setTimer() {
        timer = new Timer();
        timerTask = new TimerTask() {
            @Override
            public void run() {
                if (imageViewBitmap == null && previewView.getWidth() > 0 && previewView.getHeight() > 0) {
                    imageViewBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
                } else if (imageViewBitmap != null && imageCapture != null) {
                    takePicture();
                    //savePicture();
                    detectFaces();
                    detectEmotions();
                    drawDetections();
                }
            }
        };
        timer.scheduleAtFixedRate(timerTask, 1000, 200);
    }

    private void drawDetections() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Bitmap tempBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
                Canvas tempCanvas = new Canvas(tempBitmap);
                tempCanvas.drawBitmap(imageViewBitmap, 0, 0, null);
                //draw emotions and probabilities
                if (emotionsProbabilities != null) {
                    int max1Index = -1;
                    int max2Index = -1;
                    int max3Index = -1;

                    for (int i = 0; i < emotionsProbabilities.length; i++) {
                        if (max1Index == -1 || emotionsProbabilities[i] > emotionsProbabilities[max1Index]) {
                            max3Index = max2Index;
                            max2Index = max1Index;
                            max1Index = i;
                        } else if (max2Index == -1 || (emotionsProbabilities[i] > emotionsProbabilities[max2Index] && emotionsProbabilities[max1Index] > emotionsProbabilities[max2Index])) {
                            max3Index = max2Index;
                            max2Index = i;
                        } else if (max3Index == -1 || (emotionsProbabilities[i] > emotionsProbabilities[max3Index] && emotionsProbabilities[max2Index] > emotionsProbabilities[max3Index])) {
                            max3Index = i;
                        }
                    }
                    Paint paint1 = new Paint();
                    paint1.setColor(Color.BLACK);
                    paint1.setStrokeWidth(5);
                    paint1.setStyle(Paint.Style.FILL_AND_STROKE);
                    paint1.setTextSize(60);
                    Paint paint2 = new Paint();
                    paint2.setColor(Color.WHITE);
                    paint2.setStyle(Paint.Style.FILL);
                    paint2.setTextSize(60);
                    tempCanvas.drawText(EMOTIONS[max1Index] + ":\t\t\t" + emotionsProbabilities[max1Index], 10, 50, paint1);
                    tempCanvas.drawText(EMOTIONS[max1Index] + ":\t\t\t" + emotionsProbabilities[max1Index], 10, 50, paint2);
                    tempCanvas.drawText(EMOTIONS[max2Index] + ":\t\t\t" + emotionsProbabilities[max2Index], 10, 150, paint1);
                    tempCanvas.drawText(EMOTIONS[max2Index] + ":\t\t\t" + emotionsProbabilities[max2Index], 10, 150, paint2);
                    tempCanvas.drawText(EMOTIONS[max3Index] + ":\t\t\t" + emotionsProbabilities[max3Index], 10, 250, paint1);
                    tempCanvas.drawText(EMOTIONS[max3Index] + ":\t\t\t" + emotionsProbabilities[max3Index], 10, 250, paint2);
                }
                //draw face bounding box
                if (face != null) {
                    float widthRatio = ((float) previewView.getWidth()) / mlImage.getWidth();
                    float heightRatio = ((float) previewView.getHeight()) / mlImage.getHeight();
                    Rect origBounds = face.getBoundingBox();
                    Rect bounds = new Rect(previewView.getWidth() - (int) (origBounds.right * widthRatio), (int) (origBounds.top * heightRatio), previewView.getWidth() - (int) (origBounds.left * widthRatio), (int) (origBounds.bottom * heightRatio));

                    Paint paint1 = new Paint();
                    paint1.setColor(Color.RED);
                    paint1.setStrokeWidth(5);
                    paint1.setStyle(Paint.Style.STROKE);
                    tempCanvas.drawRect(bounds, paint1);
                    Paint paint2 = new Paint();
                    paint2.setColor(Color.WHITE);
                    paint2.setStrokeWidth(3);
                    paint2.setStyle(Paint.Style.STROKE);
                    Rect bounds2 = new Rect(bounds.left + 4, bounds.top + 4, bounds.right - 4, bounds.bottom - 4);
                    tempCanvas.drawRect(bounds2, paint2);
                }
                imageView.setImageBitmap(tempBitmap);
            }
        });
    }

    private void refreshFaceBitmap(Rect rect) {
        if (bitmap == null || rect.left < 0 || rect.top < 0 || rect.width() + Math.max(0, rect.left) > bitmap.getWidth() || rect.height() + Math.max(0, rect.top) > bitmap.getHeight()) {
            return;
        }
        //if (faceBitmap == null) {
        //    faceBitmap = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height());
        //    savePicture(faceBitmap);
        //}
        faceBitmap = Bitmap.createScaledBitmap(Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height()), 224, 224, true);
    }

    private void drawFaceBoundingBox(Face face) {
        Bitmap tempBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas tempCanvas = new Canvas(tempBitmap);
        tempCanvas.drawBitmap(imageViewBitmap, 0, 0, null);
        if (face == null) {
            imageView.setImageBitmap(tempBitmap);
            return;
        }

        float widthRatio = ((float) previewView.getWidth()) / mlImage.getWidth();
        float heightRatio = ((float) previewView.getHeight()) / mlImage.getHeight();
        Rect origBounds = face.getBoundingBox();
        Rect bounds = new Rect(previewView.getWidth() - (int) (origBounds.right * widthRatio), (int) (origBounds.top * heightRatio), previewView.getWidth() - (int) (origBounds.left * widthRatio), (int) (origBounds.bottom * heightRatio));

        Paint paint1 = new Paint();
        paint1.setColor(Color.RED);
        paint1.setStrokeWidth(5);
        paint1.setStyle(Paint.Style.STROKE);
        tempCanvas.drawRect(bounds, paint1);
        Paint paint2 = new Paint();
        paint2.setColor(Color.WHITE);
        paint2.setStrokeWidth(3);
        paint2.setStyle(Paint.Style.STROKE);
        Rect bounds2 = new Rect(bounds.left + 4, bounds.top + 4, bounds.right - 4, bounds.bottom - 4);
        tempCanvas.drawRect(bounds2, paint2);
        imageView.setImageBitmap(tempBitmap);
    }

    private void setupEmotionDetector() {
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();

        if (compatList.isDelegateSupportedOnThisDevice()) {
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegateFactory.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            interpreterOptions.addDelegate(gpuDelegate);
            Log.wtf("emotions", "GPU supported");

        } else {
            // if the GPU is not supported, run on 4 threads
            interpreterOptions.setNumThreads(4);
            Log.wtf("emotions", "GPU not supported");
        }

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
                .setMinFaceSize(0.15f)
                .enableTracking()
                .build();

        detector = FaceDetection.getClient(faceDetectorOptions);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("model_optimized.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.getStartOffset(), assetFileDescriptor.getDeclaredLength());
    }

    void bindPreview(ProcessCameraProvider cameraProvider) {
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
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        imageView = findViewById(R.id.imageView);

        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.MANAGE_EXTERNAL_STORAGE}, 2);
        }
    }

    private void setupCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            ProcessCameraProvider cameraProvider;
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
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