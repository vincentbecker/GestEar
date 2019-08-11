package ch.ethz.inf.vs.gestear.datacollection.imu;

import java.util.List;

public class LerpResizer {

    private static final float EPSILON = 0.000001f;
    private static LerpResizer instance = null;

    public static LerpResizer getInstance() {
        if (instance == null) {
            instance = new LerpResizer();
        }
        return instance;
    }

    private LerpResizer() {

    }

    public float[][][] resizeSequence(List<IMUEntry> data, int fs, double startTime, float dt, int sequenceLength) {
        double endTime = startTime + dt;

        boolean addedAtStart = false;
        if (startTime < data.get(0).getTimestamp()) {
            data.add(0, new IMUEntry(startTime, data.get(0).getValues()));
            addedAtStart = true;
        }
        boolean addedAtEnd = false;
        if (data.get(data.size() - 1).getTimestamp() < endTime) {
            data.add(new IMUEntry(endTime, data.get(data.size() - 1).getValues()));
            addedAtEnd = true;
        }

        int m = (int) (dt * fs);
        //if (m % 2 == 1) {
        //    m++;
        //}
        float timeStep = 1.0f / fs;
        int windowLength = m / sequenceLength;
        float[][][] result = new float[sequenceLength][windowLength][3];

        double t = startTime;
        int currentIndex = 0;
        for (int i = 0; i < m; i++) {
            // Search the two timestamps around t and interpolate if necessary
            for (int j = currentIndex; j < data.size(); j++) {
                IMUEntry entry = data.get(j);
                double timestamp = entry.getTimestamp();
                if (Math.abs(timestamp - t) < EPSILON) {
                    // This sample has nearly the same timestamp
                    result[i / windowLength][i % windowLength] = entry.getValues();
                    currentIndex = j + 1;
                    break;
                } else if (timestamp > t) {
                    // Search for entry with timestamp smaller than t
                    int previousIndex = j - 1;
                    while (data.get(previousIndex).getTimestamp() > t) {
                        previousIndex--;
                    }
                    IMUEntry previousEntry = data.get(previousIndex);
                    result[i / windowLength][i % windowLength] = interpolate(t, previousEntry.getTimestamp(), previousEntry.getValues(), entry.getTimestamp(), entry.getValues());
                    currentIndex = j;
                    break;
                }
            }
            t += timeStep;
        }

        if (addedAtStart) {
            data.remove(0);
        }
        if (addedAtEnd) {
            data.remove(data.size() - 1);
        }

        return result;
    }

    public float[][] resizeWindow(List<IMUEntry> data, int fs, double startTime, float dt) {
        double endTime = startTime + dt;

        boolean addedAtStart = false;
        if (startTime < data.get(0).getTimestamp()) {
            data.add(0, new IMUEntry(startTime, data.get(0).getValues()));
            addedAtStart = true;
        }
        boolean addedAtEnd = false;
        if (data.get(data.size() - 1).getTimestamp() < endTime) {
            data.add(new IMUEntry(endTime, data.get(data.size() - 1).getValues()));
            addedAtEnd = true;
        }

        int m = (int) (dt * fs);
        //if (m % 2 == 1) {
        //    m++;
        //}
        float timeStep = 1.0f / fs;
        float[][] result = new float[m][3];

        double t = startTime;
        int currentIndex = 0;
        for (int i = 0; i < m; i++) {
            // Search the two timestamps around t and interpolate if necessary
            for (int j = currentIndex; j < data.size(); j++) {
                IMUEntry entry = data.get(j);
                double timestamp = entry.getTimestamp();
                if (Math.abs(timestamp - t) < EPSILON) {
                    // This sample has nearly the same timestamp
                    result[i] = entry.getValues();
                    currentIndex = j + 1;
                    break;
                } else if (timestamp > t) {
                    // Search for entry with timestamp smaller than t
                    int previousIndex = j - 1;
                    while (data.get(previousIndex).getTimestamp() > t) {
                        previousIndex--;
                    }
                    IMUEntry previousEntry = data.get(previousIndex);
                    result[i] = interpolate(t, previousEntry.getTimestamp(), previousEntry.getValues(), entry.getTimestamp(), entry.getValues());
                    currentIndex = j;
                    break;
                }
            }
            t += timeStep;
        }

        if (addedAtStart) {
            data.remove(0);
        }
        if (addedAtEnd) {
            data.remove(data.size() - 1);
        }

        return result;
    }

    private float[] interpolate(double t, double t1, float[] x1, double t2, float[] x2) {
        double dt = t2 - t1;
        float w1 = (float) ((t2 - t) / dt);
        float w2 = (float) ((t - t1) / dt);
        float[] x = new float[x1.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = w1 * x1[i] + w2 * x2[i];
        }
        return x;
    }
}
