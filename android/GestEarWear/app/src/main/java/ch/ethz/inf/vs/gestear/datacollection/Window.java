package ch.ethz.inf.vs.gestear.datacollection;

public class Window {
    private float[] fftData;
    private float[][] gyroData;
    private float[][] linAccelData;
    private float totalMagnitude;

    public float[] getFFTData() {
        return fftData;
    }

    public float[][] getGyroData() {
        return gyroData;
    }

    public float[][] getLinAccelData() {
        return linAccelData;
    }

    public float getTotalMagnitude() {
        return totalMagnitude;
    }

    Window(float[] fftData, float[][] gyroData, float[][] linAccelData) {
        this.fftData = fftData;
        this.gyroData = gyroData;
        this.linAccelData = linAccelData;
        for (float f : fftData) {
            totalMagnitude += f;
        }
    }
}
