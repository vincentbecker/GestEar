package ch.ethz.inf.vs.gestear.datacollection;

public class Sequence {

    private float[][] fftData;
    private float[][][] gyroData;
    private float[][][] linAccelData;
    private float totalMagnitude;

    public float[][] getFFTData() {
        return fftData;
    }

    public float[][][] getGyroData() {
        return gyroData;
    }

    public float[][][] getLinAccelData() {
        return linAccelData;
    }

    public float getTotalMagnitude() {
        return totalMagnitude;
    }

    Sequence(float[][] fftData, float[][][] gyroData, float[][][] linAccelData) {
        this.fftData = fftData;
        this.gyroData = gyroData;
        this.linAccelData = linAccelData;
    }

    Sequence(Window[] windows) {
        fftData = new float[Settings.SEQUENCE_LENGTH][Settings.Audio.KEEP_NUMBER_OF_BINS];
        gyroData = new float[Settings.SEQUENCE_LENGTH][Settings.Sensor.Gyroscope.SAMPLES_PER_WINDOW][3];
        linAccelData = new float[Settings.SEQUENCE_LENGTH][Settings.Sensor.LinearAcceleration.SAMPLES_PER_WINDOW][3];
        for (int i = 0; i < Settings.SEQUENCE_LENGTH; i++) {
            fftData[i] = windows[i].getFFTData();
            gyroData[i] = windows[i].getGyroData();
            linAccelData[i] = windows[i].getLinAccelData();
            totalMagnitude = windows[i].getTotalMagnitude();
        }
    }

    public interface SequenceCallback {
        void onNewSequence(Sequence sequence);
    }
}
