package ch.ethz.inf.vs.gestear.datacollection;

import android.util.Log;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import ch.ethz.inf.vs.gestear.MainActivity;
import ch.ethz.inf.vs.gestear.datacollection.audio.AudioRecorder;
import ch.ethz.inf.vs.gestear.datacollection.imu.IMUEntry;
import ch.ethz.inf.vs.gestear.datacollection.imu.IMURecorder;

public class DataCollector2 implements AudioRecorder.AudioRecorderListener {

    private static final String TAG = "DATA_COLLECTOR";

    private Sequence.SequenceCallback callback;
    private AudioRecorder audioRecorder;
    private IMURecorder imuRecorder;
    private ScheduledExecutorService scheduledExecutorService;
    private SequenceHandler sequenceHandler;

    public DataCollector2(MainActivity activity) {
        callback = activity;
        audioRecorder = new AudioRecorder(this);
        imuRecorder = new IMURecorder(activity);
        sequenceHandler = new SequenceHandler();
    }

    public void start() {
        // start audio recorder and wait for callback until recording started
        audioRecorder.startRecording();
    }

    @Override
    public void onAudioRecorderStarted() {
        // audio recorder started, start data collection
        imuRecorder.startDataCollection();

        // regularly get data
        final long delay = (long) (1000 * Settings.RECORDING_DURATION);

        scheduledExecutorService = Executors.newScheduledThreadPool(1);
        scheduledExecutorService.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                processData();
            }
        }, delay, delay, TimeUnit.MILLISECONDS);
    }

    private void processData() {
        Log.d(TAG, "Process data");
        long startProcess = System.nanoTime();
        float[] audioData = audioRecorder.getData();
        List<List<IMUEntry>> imuData = imuRecorder.getData();
        Sequence sequence = sequenceHandler.addData(audioData, imuData.get(0), imuData.get(1));
        long endProcess = System.nanoTime();
        MainActivity.timePreprocess += (endProcess - startProcess);
        if (sequence != null) {
            callback.onNewSequence(sequence);
        }
    }

    public void stop() {
        Log.d(TAG, "Stopping recording");
        if (scheduledExecutorService != null) {
            scheduledExecutorService.shutdownNow();
        }
        if (audioRecorder != null) {
            audioRecorder.stopRecording();
        }
        if (imuRecorder != null) {
            imuRecorder.stopDataCollection();
        }
    }
}
