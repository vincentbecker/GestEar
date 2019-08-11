package ch.ethz.inf.vs.gestear.datacollection.imu;

public class IMUEntry {

    private double timestamp;
    private float[] values;

    public double getTimestamp() {
        return timestamp;
    }

    public float[] getValues() {
        return values;
    }

    public IMUEntry(double timestamp, float[] values) {
        this.timestamp = timestamp;
        this.values = values;
    }
}
