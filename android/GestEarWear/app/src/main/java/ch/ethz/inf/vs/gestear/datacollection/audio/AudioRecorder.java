package ch.ethz.inf.vs.gestear.datacollection.audio;

import android.media.AudioRecord;
import android.os.Process;
import android.util.Log;

import ch.ethz.inf.vs.gestear.datacollection.Settings;

public class AudioRecorder extends Thread implements AudioRecord.OnRecordPositionUpdateListener {

    private static final String TAG = "AUDIO_RECORDER";
    private AudioRecord audioRecord;
    private AudioRecorderListener listener;

    public AudioRecorder(AudioRecorderListener listener) {
        super("AudioRecorder");
        this.listener = listener;
    }

    public void startRecording() {
        start();
    }

    @Override
    public void run() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        audioRecord = new AudioRecord(
                Settings.Audio.AUDIO_SOURCE,
                Settings.Audio.SAMPLE_RATE,
                Settings.Audio.CHANNEL_IN_CONFIG,
                Settings.Audio.ENCODING,
                Settings.Audio.BUFFER_SIZE_IN_BYTES
        );
        // add listener...
        audioRecord.setRecordPositionUpdateListener(this);
        // ...to get notified when recording actually starts
        audioRecord.setNotificationMarkerPosition(1);
        // and startDataCollection recording
        audioRecord.startRecording();
    }

    @Override
    public void onMarkerReached(AudioRecord audioRecord) {
        if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
            Log.d(AudioRecorder.class.getSimpleName(), "Started recording.");
            listener.onAudioRecorderStarted();
        }
    }

    public float[] getData() {
        byte[] data = new byte[Settings.Audio.BUFFER_SIZE_IN_BYTES];
        if (audioRecord != null) {
            long startReading = System.nanoTime();
            audioRecord.read(data, 0, data.length);
            long endReading = System.nanoTime();
            // Log.d(TAG, "Read data: " + (endReading - startReading) / 1000000.0 + " ms");
        }
        return AudioRecorder.convertByteToFloat(data);
    }

    public void stopRecording() {
        Log.d(AudioRecorder.class.getSimpleName(), "Stopped recording.");
        if (audioRecord != null) {
            if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                audioRecord.stop();
            }
            audioRecord.release();
            audioRecord = null;
        }
        try {
            join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static float[] convertByteToFloat(byte[] bytes) {
        int bytesPerSample = Settings.Audio.BYTES_PER_SAMPLE;
        int numberOfSamples = bytes.length / bytesPerSample;

        float[] floats = new float[numberOfSamples];
        for (int i = 0; i < numberOfSamples; i++) {
            for (int b = 0; b < bytesPerSample; b++) {
                int v = bytes[bytesPerSample*i+b];
                if (b < bytesPerSample - 1 || bytesPerSample == 1) {
                    v &= 0xFF;
                }
                floats[i] += v << (b * 8);
            }
            floats[i] /= 32768f;
        }
        return floats;
    }

    @Override
    public void onPeriodicNotification(AudioRecord audioRecord) {

    }

    public interface AudioRecorderListener {
        void onAudioRecorderStarted();
    }
}
