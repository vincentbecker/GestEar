package ch.ethz.inf.vs.gestear.datacollection.audio;

import com.paramsen.noise.Noise;
import com.paramsen.noise.NoiseOptimized;
import static ch.ethz.inf.vs.gestear.datacollection.Settings.Audio.FFT_LENGTH;
import static ch.ethz.inf.vs.gestear.datacollection.Settings.Audio.KEEP_NUMBER_OF_BINS;
import static ch.ethz.inf.vs.gestear.datacollection.Settings.Audio.NUMBER_OF_BINS;

public class FFTNoise {

    private static final String TAG = "FFT_CALCULATOR";
    private final NoiseOptimized noise;
    private final float[] hannWindow;

    private static FFTNoise instance = null;

    public static void initialize(int samplesPerWindow) {
        instance = new FFTNoise(samplesPerWindow);
    }

    public static FFTNoise getInstance() {
        return instance;
    }

    private FFTNoise(int samplesPerWindow) {
        noise = Noise.real().optimized().init(FFT_LENGTH, true);

        hannWindow = new float[samplesPerWindow];
        for (int i = 0; i < hannWindow.length; i++) {
            hannWindow[i] = (float) (0.5 * (1 - Math.cos(2 * Math.PI * i / (hannWindow.length - 1))));
        }
    }

    private void hann(float[] audio) {
        for (int i = 0; i < hannWindow.length; i++) {
            audio[i] *= hannWindow[i];
        }
    }

    public float[] fft(float[] audio) {
        float[] input = audio.clone();
        long s = System.nanoTime();
        hann(input);
        long e = System.nanoTime();
        //Log.d(TAG, "Timecost for hann: " + (e - s) / 1000000f + " ms");
        //MainActivity.hannRuntimes.add(e - s);

        float[] zeroPadded = new float[FFT_LENGTH];
        System.arraycopy(input, 0, zeroPadded, 0, input.length);
        // fft
        s = System.nanoTime();
        float[] output = noise.fft(zeroPadded);
        e = System.nanoTime();
        //Log.d(TAG, "Timecost for fft calculation: " + (e - s) / 1000000f + " ms");
        //MainActivity.fftRuntimes.add(e - s);

        // the physical layout of the output data is as follows:
        // Re[0], Re[n/2], Re[1], Im[1], Re[2], Im[2], ...

        // calculate magnitudes
        int l = output.length / 2;
        float[] magnitudes = new float[l + 1];
        float normFactor = 1f / l;
        magnitudes[0] = normFactor * Math.abs(output[0]) / 2;
        for (int i = 1; i < magnitudes.length - 1; i++) {
            magnitudes[i] = normFactor * (float) Math.sqrt(output[2 * i] * output[2 * i] + output[2 * i + 1] * output[2 * i + 1]);
        }
        magnitudes[magnitudes.length - 1] = normFactor * Math.abs(output[1]) / 2;

        // reduce to less bins
        int binRatio = magnitudes.length / NUMBER_OF_BINS;
        float[] bins = new float[KEEP_NUMBER_OF_BINS];
        for (int i = 0; i < bins.length; i++) {
            for (int j = 0; j < binRatio; j++) {
                bins[i] += magnitudes[binRatio * i + j];
            }
        }
        return bins;
    }
}
