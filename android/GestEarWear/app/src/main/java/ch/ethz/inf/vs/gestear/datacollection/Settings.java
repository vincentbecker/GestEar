package ch.ethz.inf.vs.gestear.datacollection;

import android.media.AudioFormat;
import android.media.MediaRecorder;

public final class Settings {

    // the following durations are in seconds
    static final float WINDOW_DURATION = 0.3f;
    static final float RECORDING_DURATION = 0.15f;
    static final int SEQUENCE_LENGTH = 2;

    private Settings() {}

    public static final class Audio {

        public static final int AUDIO_SOURCE = MediaRecorder.AudioSource.MIC;
        public static final int SAMPLE_RATE = 22050;
        public static final int SAMPLES_PER_WINDOW = (int) (WINDOW_DURATION * SAMPLE_RATE);
        static final int SAMPLES_PER_RECORDING = (int) (RECORDING_DURATION * SAMPLE_RATE);
        public static final int FFT_LENGTH = 8192;
        static final int NUM_CHANNELS = 1;
        public static final int CHANNEL_IN_CONFIG = NUM_CHANNELS == 1
                        ? AudioFormat.CHANNEL_IN_MONO
                        : AudioFormat.CHANNEL_IN_STEREO;
        public static final int CHANNEL_OUT_CONFIG = NUM_CHANNELS == 1
                        ? AudioFormat.CHANNEL_OUT_MONO
                        : AudioFormat.CHANNEL_OUT_STEREO;
        public static final int ENCODING = AudioFormat.ENCODING_PCM_16BIT;
        public static final int BYTES_PER_SAMPLE =
                ENCODING == AudioFormat.ENCODING_PCM_8BIT ? 1
                        : ENCODING == AudioFormat.ENCODING_PCM_16BIT ? 2
                        : ENCODING == AudioFormat.ENCODING_PCM_FLOAT ? 4
                        : 0;

        static final int BUFFER_SIZE_IN_SHORTS = NUM_CHANNELS * SAMPLES_PER_RECORDING;
        public static final int BUFFER_SIZE_IN_BYTES = BYTES_PER_SAMPLE * BUFFER_SIZE_IN_SHORTS;
        public static final int NUMBER_OF_BINS = 128;
        public static final int KEEP_NUMBER_OF_BINS = 96;

        public static final float MINIMUM_MAGNITUDE = 0.15f;

        private Audio() {}
    }

    public static final class Sensor {

        public static final class Gyroscope {

            public static final int SAMPLE_RATE = 200;
            public static final int SAMPLES_PER_WINDOW = (int) (SAMPLE_RATE * WINDOW_DURATION);

            private Gyroscope() {}
        }

        public static final class LinearAcceleration {

            public static final int SAMPLE_RATE = 200;
            public static final int SAMPLES_PER_WINDOW = (int) (SAMPLE_RATE * WINDOW_DURATION);

            private LinearAcceleration() {}
        }
    }
}
