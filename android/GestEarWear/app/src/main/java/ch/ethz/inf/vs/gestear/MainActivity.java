package ch.ethz.inf.vs.gestear;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.wearable.activity.WearableActivity;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import com.jjoe64.graphview.DefaultLabelFormatter;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.GridLabelRenderer;
import com.jjoe64.graphview.series.BarGraphSeries;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import java.io.IOException;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import ch.ethz.inf.vs.gestear.datacollection.DataCollector2;
import ch.ethz.inf.vs.gestear.datacollection.Sequence;
import ch.ethz.inf.vs.gestear.datacollection.Settings;

public class MainActivity extends WearableActivity implements Sequence.SequenceCallback {

    private static final String TAG = "MAIN_ACTIVITY";
    private static final int PERMISSIONS_REQUEST_RECORD_AUDIO = 5345;

    // Recording attributed
    DataCollector2 mDataCollector;
    Queue<Sequence> mSequenceQueue;

    // Classifier attributes
    private HandlerThread backgroundThread;
    private static final String HANDLE_THREAD_NAME = "BackgroundThread";
    private Handler backgroundHandler;
    private final Object lock = new Object();
    private boolean runClassifier = false;
    private Classifier classifier;
    private int toastCounter = 0;
    private PostProcessor2 postProcessor;

    private DataPoint[] mFFTDataPoints;
    private DataPoint[] mGyroXDataPoints;
    private DataPoint[] mGyroYDataPoints;
    private DataPoint[] mGyroZDataPoints;
    private BarGraphSeries<DataPoint> mFFTSeries;
    private LineGraphSeries<DataPoint> mGyroXSeries;
    private LineGraphSeries<DataPoint> mGyroYSeries;
    private LineGraphSeries<DataPoint> mGyroZSeries;
    private TextView mTextView;

    public static int sequenceCount = 0;
    public static long timeClassifySample = 0;
    public static long timePreprocess = 0;
    public static long timeFFT = 0;
    public static long timeLerpResize = 0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Enables Always-on
        setAmbientEnabled();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, PERMISSIONS_REQUEST_RECORD_AUDIO);
        }

        mDataCollector = new DataCollector2(this);
        mSequenceQueue = new ConcurrentLinkedQueue<>();

        try {
            classifier = new Classifier(this);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize a classifier.", e);
        }
        postProcessor = new PostProcessor2();

        GraphView mGraphView = findViewById(R.id.graph);
        mFFTDataPoints = new DataPoint[Settings.Audio.KEEP_NUMBER_OF_BINS];
        for (int i = 0; i < mFFTDataPoints.length; i++) {
            mFFTDataPoints[i] = new DataPoint(i, i);
        }
        mFFTSeries = new BarGraphSeries<>(mFFTDataPoints);
        mFFTSeries.setSpacing(0);
        mGraphView.addSeries(mFFTSeries);
        mGraphView.getViewport().setYAxisBoundsManual(true);
        mGraphView.getViewport().setMinY(0);
        mGraphView.getViewport().setMaxY(0.02);

        mGyroXDataPoints = new DataPoint[Settings.Sensor.Gyroscope.SAMPLES_PER_WINDOW];
        mGyroYDataPoints = new DataPoint[Settings.Sensor.Gyroscope.SAMPLES_PER_WINDOW];
        mGyroZDataPoints = new DataPoint[Settings.Sensor.Gyroscope.SAMPLES_PER_WINDOW];
        for (int i = 0; i < Settings.Sensor.Gyroscope.SAMPLES_PER_WINDOW; i++) {
            mGyroXDataPoints[i] = new DataPoint(i, i);
            mGyroYDataPoints[i] = new DataPoint(i, i);
            mGyroZDataPoints[i] = new DataPoint(i, i);
        }
        mGyroXSeries = new LineGraphSeries<>(mGyroXDataPoints);
        mGyroXSeries.setColor(Color.RED);
        mGraphView.getSecondScale().addSeries(mGyroXSeries);
        mGyroYSeries = new LineGraphSeries<>(mGyroYDataPoints);
        mGyroYSeries.setColor(Color.YELLOW);
        mGraphView.getSecondScale().addSeries(mGyroYSeries);
        mGyroZSeries = new LineGraphSeries<>(mGyroZDataPoints);
        mGyroZSeries.setColor(Color.GREEN);
        mGraphView.getSecondScale().addSeries(mGyroZSeries);
        mGraphView.getSecondScale().setMinY(-10);
        mGraphView.getSecondScale().setMaxY(10);
        mGraphView.getSecondScale().setLabelFormatter(new DefaultLabelFormatter() {
            @Override
            public String formatLabel(double value, boolean isValueX) {
                return "";
            }
        });

        GridLabelRenderer renderer = mGraphView.getGridLabelRenderer();
        renderer.setHorizontalLabelsVisible(false);
        renderer.setVerticalLabelsVisible(false);
        renderer.setGridStyle(GridLabelRenderer.GridStyle.NONE);
        mTextView = findViewById(R.id.text);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST_RECORD_AUDIO: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    startRecordingAndClassifying();
                } else {
                    showToast("Cannot be used without this permission");
                    // Stop app
                    finish();
                }
                break;
            }
        }
    }

    private void startRecordingAndClassifying() {
        // Create DataCollector and start recording
        startBackgroundThread();
        mDataCollector.start();
    }

    private void updateGraph(float[] fft, float[][] gyro, float[][] linAccel) {
        for (int i = 0; i < mFFTDataPoints.length; i++) {
            mFFTDataPoints[i] = new DataPoint(i, fft[i]);
        }
        mFFTSeries.resetData(mFFTDataPoints);

        for (int i = 0; i < mGyroXDataPoints.length; i++) {
            mGyroXDataPoints[i] = new DataPoint(i, gyro[i][0]);
            mGyroYDataPoints[i] = new DataPoint(i, gyro[i][1]);
            mGyroZDataPoints[i] = new DataPoint(i, gyro[i][2]);
        }
        mGyroXSeries.resetData(mGyroXDataPoints);
        mGyroYSeries.resetData(mGyroYDataPoints);
        mGyroZSeries.resetData(mGyroZDataPoints);
    }

    /**
     * Shows a {@link Toast} on the UI thread for the classification results.
     *
     * @param s The message to show
     */
    private void showToast(String s) {
        SpannableStringBuilder builder = new SpannableStringBuilder();
        SpannableString str1 = new SpannableString(s);
        builder.append(str1);
        showToast(builder);
    }

    private void showToast(final SpannableStringBuilder builder) {
        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mTextView.setText(builder, TextView.BufferType.SPANNABLE);
                    }
                });
    }

    @Override
    public void onResume() {
        super.onResume();
        startRecordingAndClassifying();
    }

    @Override
    public void onPause() {
        mDataCollector.stop();
        stopBackgroundThread();
        super.onPause();
    }

    @Override
    public void onDestroy() {
        classifier.close();
        Log.d(TAG, "Sequence count: " + sequenceCount);
        Log.d(TAG, "Time calculateFFT(): " + timeFFT  / 1000000000.0 / sequenceCount);
        Log.d(TAG, "Time 2x lerpResize(): " + timeLerpResize  / 1000000000.0 / sequenceCount);
        super.onDestroy();
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        synchronized (lock) {
            runClassifier = true;
        }
        // Will be posted when new data is available
        backgroundHandler.post(periodicClassify);
    }

    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
            synchronized (lock) {
                runClassifier = false;
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted when stopping background thread", e);
        }
    }

    @Override
    public void onNewSequence(final Sequence sequence) {
        // Update the graph
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                float[] fft = sequence.getFFTData()[1];
                float[][] gyro = sequence.getGyroData()[1];
                float[][] linAccel = sequence.getLinAccelData()[1];
                updateGraph(fft, gyro, linAccel);
            }
        });
        float magnitudeSum = sequence.getTotalMagnitude();
        String s = "Msum: " + magnitudeSum;
        //showToast(s);
        Log.d(TAG, s);
        if (magnitudeSum >= Settings.Audio.MINIMUM_MAGNITUDE) {
            // Log.d(TAG, "Adding new sequence");
            mSequenceQueue.add(sequence);
        } else {
            // Inform the post processor that a time slot has passed
            postProcess(new RecognitionResult(0, 0));
        }
        sequenceCount++;
    }

    private Runnable periodicClassify =
            new Runnable() {
                @Override
                public void run() {
                    synchronized (lock) {
                        if (!mSequenceQueue.isEmpty()) {
                            if (runClassifier) {
                                classifySample(mSequenceQueue.poll());
                            }
                        }
                        backgroundHandler.post(periodicClassify);
                    }
                }
            };

    /**
     * Classifies a sample.
     */
    private void classifySample(Sequence sequence) {
        long start = System.nanoTime();
        SpannableStringBuilder textToShow = new SpannableStringBuilder();
        RecognitionResult result = classifier.classifySample(sequence, textToShow);
        result.setFftMagnitude(sequence.getTotalMagnitude());
        // Log.d(TAG, "Gesture index: " + gestureIndex);
        postProcess(result);
        long end = System.nanoTime();
        timeClassifySample += (end - start);
        Log.d(TAG, "Time to classify sample " + timeClassifySample);
    }

    private void postProcess(RecognitionResult result) {
        Log.d(TAG, "Gesture classified: " + PostProcessor2.Gesture.values()[result.getLabel()].name());
        SpannableStringBuilder textToShow = new SpannableStringBuilder();
        PostProcessor2.Gesture gesture = postProcessor.performedGesture(result);
        Log.d(TAG, "Detected gesture: " + gesture.name());
        if (gesture != PostProcessor2.Gesture.NULL) {
            textToShow.append(new SpannableString(gesture.name()));
            showToast(textToShow);
            toastCounter = 0;
        } else {
            if (toastCounter > 2) {
                showToast("");
            }
        }
        toastCounter++;
    }
}
