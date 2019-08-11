package ch.ethz.inf.vs.gestear.datacollection;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.inf.vs.gestear.MainActivity;
import ch.ethz.inf.vs.gestear.datacollection.audio.FFTNoise;
import ch.ethz.inf.vs.gestear.datacollection.imu.IMUEntry;
import ch.ethz.inf.vs.gestear.datacollection.imu.LerpResizer;

class SequenceHandler {
    private static final String TAG = "SEQUENCE_HANDLER";
    private static final int SUBWINDOWS_PER_WINDOW = (int) (Settings.WINDOW_DURATION / Settings.RECORDING_DURATION);
    private static final int SAMPLES_PER_WINDOW = SUBWINDOWS_PER_WINDOW * Settings.Audio.SAMPLES_PER_RECORDING;
    private List<float[]> audioList = new ArrayList<>();
    private List<List<IMUEntry>> gyroList = new ArrayList<>();
    private List<List<IMUEntry>> linAccelList = new ArrayList<>();
    private List<List<Window>> queues = new ArrayList<>();
    private int queueIndex = 0;

    SequenceHandler() {
        FFTNoise.initialize(SAMPLES_PER_WINDOW);
        // Adding two queues
        queues.add(new ArrayList<Window>(2));
        queues.add(new ArrayList<Window>(2));
    }

    Sequence addData(float[] audio, List<IMUEntry> gyro, List<IMUEntry> linAccel) {
        if (gyro.size() == 0 || linAccel.size() == 0) {
            return null;
        }
        audioList.add(audio);
        gyroList.add(gyro);
        linAccelList.add(linAccel);
        // long startTimeTotal = SystemClock.uptimeMillis();
        if (audioList.size() >= SUBWINDOWS_PER_WINDOW) {
            float[] audioWindow = new float[SAMPLES_PER_WINDOW];
            List<IMUEntry> gyroCombined = new ArrayList<>();
            List<IMUEntry> linAccelCombined = new ArrayList<>();
            for (int i = 0; i < SUBWINDOWS_PER_WINDOW; i++) {
                System.arraycopy(audioList.get(i), 0, audioWindow, i * Settings.Audio.SAMPLES_PER_RECORDING, Settings.Audio.SAMPLES_PER_RECORDING);
                gyroCombined.addAll(gyroList.get(i));
                linAccelCombined.addAll(linAccelList.get(i));
            }
            long startFFT = System.nanoTime();
            float[] fft = FFTNoise.getInstance().fft(audioWindow);
            long endFFT = System.nanoTime();
            MainActivity.timeFFT += (endFFT - startFFT);
            double startTimestamp = Math.min(gyroCombined.get(0).getTimestamp(), linAccelCombined.get(0).getTimestamp());
            long startLerpResize = System.nanoTime();
            float[][] gyroResized = preprocessIMUData(gyroCombined, Settings.Sensor.Gyroscope.SAMPLE_RATE, startTimestamp);
            float[][] linAccelResized = preprocessIMUData(linAccelCombined, Settings.Sensor.LinearAcceleration.SAMPLE_RATE, startTimestamp);
            long endLerpResize = System.nanoTime();
            MainActivity.timeLerpResize += (endLerpResize - startLerpResize);

            // Removing oldest subwindow
            audioList.remove(0);
            gyroList.remove(0);
            linAccelList.remove(0);

            List<Window> currentQueue = queues.get(queueIndex);
            currentQueue.add(new Window(fft, gyroResized, linAccelResized));
            queueIndex = (queueIndex + 1) % 2;
            if (currentQueue.size() >= Settings.SEQUENCE_LENGTH) {
                Window[] windows = new Window[2];
                windows[0] = currentQueue.remove(0);
                windows[1] = currentQueue.get(0);
                return new Sequence(windows);
            }
        }
        return null;
    }

    private float[][] preprocessIMUData(List<IMUEntry> imuEntries, int samplingRate, double startTimestamp) {
        return LerpResizer.getInstance().resizeWindow(imuEntries, samplingRate, startTimestamp, Settings.WINDOW_DURATION);
    }

}
