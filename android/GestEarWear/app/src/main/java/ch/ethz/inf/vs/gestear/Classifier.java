package ch.ethz.inf.vs.gestear;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.ethz.inf.vs.gestear.datacollection.Sequence;
import ch.ethz.inf.vs.gestear.datacollection.Settings;

/**
 * Classifies images with Tensorflow Lite.
 */
class Classifier {

    private static final float PROB_THRESHOLD = 0.9f;

    /**
     * Tag for the {@link Log}.
     */
    private static final String TAG = "Classifier";

    private static final String modelPath = "path/to/tflite/model.tflite";
    private static final String labelPath = "labels.txt";

    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int SEQUENCE_LENGTH = 2;
    private static final int LENGTH_FFT = Settings.Audio.KEEP_NUMBER_OF_BINS;
    private static final int NUM_AXES = 3;
    private static final int LENGTH_GYRO = 60;
    private static final int LENGTH_LINACCEL = 60;

    private static final int FLOAT_SIZE = 4;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    private Interpreter tflite;

    /**
     * Labels corresponding to the output of the model.
     */
    private List<String> labelList;

    /**
     * ByteBuffers to hold data, to be feed into Tensorflow Lite as inputs.
     */
    private ByteBuffer fftData;
    private ByteBuffer gyroData;
    private ByteBuffer linAccelData;

    /**
     * Initializes an {@code ImageClassifier}.
     */
    Classifier(Activity activity) throws IOException {
        tfliteModel = loadModelFile(activity);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        tflite.resizeInput(0, new int[]{2, LENGTH_FFT});
        tflite.resizeInput(1, new int[]{2, LENGTH_GYRO, 3});
        tflite.resizeInput(2, new int[]{2, LENGTH_LINACCEL, 3});
        labelList = loadLabelList(activity);
        fftData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * SEQUENCE_LENGTH * LENGTH_FFT * FLOAT_SIZE);
        fftData.order(ByteOrder.nativeOrder());
        gyroData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * SEQUENCE_LENGTH * LENGTH_GYRO * NUM_AXES * FLOAT_SIZE);
        gyroData.order(ByteOrder.nativeOrder());
        linAccelData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * SEQUENCE_LENGTH * LENGTH_LINACCEL * NUM_AXES * FLOAT_SIZE);
        linAccelData.order(ByteOrder.nativeOrder());
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    /**
     * Classifies a frame from the preview stream.
     */
    RecognitionResult classifySample(Sequence sequence, SpannableStringBuilder builder) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            builder.append(new SpannableString("Uninitialized Classifier."));
        }
        convertSequenceToByteBuffer(sequence);
        ByteBuffer[] inputs = {fftData, gyroData, linAccelData};
        //ByteBuffer[] inputs = {gyroData, linAccelData};
        float[][] gestureDetectionProb = new float[1][2];
        float[][] gestureTypeLabelProbs = new float[1][labelList.size()];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, gestureDetectionProb);
        outputs.put(1, gestureTypeLabelProbs);
        //outputs.put(0, gestureTypeLabelProbs);

        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long endTime = SystemClock.uptimeMillis();
        // Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime) + " ms");
        boolean gesture_detected = false;
        if (gestureDetectionProb[0][0] < gestureDetectionProb[0][1]) {
            gesture_detected = true;
        }
        StringBuilder sb = new StringBuilder();
        sb.append("GD: " + gestureDetectionProb[0][0] + ", " + gestureDetectionProb[0][1] + "; ");
        sb.append("GT: ");
        for (float f : gestureTypeLabelProbs[0]) {
            sb.append(f);
            sb.append(", ");
        }
        Log.d(TAG, sb.toString());
        if (!gesture_detected) {
            return new RecognitionResult(0, 0);
        } else {
            // Get the matching label
            int gestureIndex = 0;
            float probability  =0f;
            for (int i = 0; i < gestureTypeLabelProbs[0].length; i++) {
                probability = gestureTypeLabelProbs[0][i];
                if (probability >= PROB_THRESHOLD) {
                    gestureIndex = i;
                    break;
                }
            }
            return new RecognitionResult(gestureIndex, probability);
        }
    }

    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    /**
     * Closes tflite to release resources.
     */
    void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private void convertSequenceToByteBuffer(Sequence sequence) {
        fftData.rewind();
        gyroData.rewind();
        linAccelData.rewind();
        long startTime = SystemClock.uptimeMillis();
        float[][] fftSample = sequence.getFFTData();
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            for (int j = 0; j < LENGTH_FFT; j++) {
                fftData.putFloat(fftSample[i][j]);
            }
        }
        float[][][] gyroSample = sequence.getGyroData();
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            for (int j = 0; j < LENGTH_GYRO; j++) {
                for (int k = 0; k < NUM_AXES; k++) {
                    gyroData.putFloat(gyroSample[i][j][k]);
                }
            }
        }
        float[][][] linAccelSample = sequence.getLinAccelData();
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            for (int j = 0; j < LENGTH_LINACCEL; j++) {
                for (int k = 0; k < NUM_AXES; k++) {
                    linAccelData.putFloat(linAccelSample[i][j][k]);
                }
            }
        }
        long endTime = SystemClock.uptimeMillis();
        // Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime) + " ms");
    }
}