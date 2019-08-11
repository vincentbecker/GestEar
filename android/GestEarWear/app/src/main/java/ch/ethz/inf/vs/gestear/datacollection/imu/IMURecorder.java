package ch.ethz.inf.vs.gestear.datacollection.imu;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Process;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.inf.vs.gestear.datacollection.Settings;

public class IMURecorder extends Thread implements SensorEventListener {

    private Context context;
    private SensorManager sensorManager;
    private List<IMUEntry> gyroData = new ArrayList<>();
    private List<IMUEntry> linAccelData = new ArrayList<>();
    private long referenceTimestamp;
    private boolean started;

    public IMURecorder(Context context) {
        super("SensorDataCollector");
        this.context = context;
        createEntryLists();
        start();
    }

    @Override
    public void run() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        registerListeners();
    }

    public void startDataCollection() {
        referenceTimestamp = 0;
        started = true;
    }

    public List<List<IMUEntry>>  getData() {
        List<List<IMUEntry>> imuData = new ArrayList<>(2);
        imuData.add(gyroData);
        imuData.add(linAccelData);
        createEntryLists();
        return imuData;
    }

    public void stopDataCollection() {
        started = false;
        unregisterListeners();

        try {
            join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void registerListeners() {
        sensorManager = ((SensorManager) context.getSystemService(Context.SENSOR_SERVICE));
        if (sensorManager != null) {
            registerListener(Sensor.TYPE_GYROSCOPE);
            registerListener(Sensor.TYPE_LINEAR_ACCELERATION);
        }
    }

    private void registerListener(int sensorType) {
        Sensor sensor = sensorManager.getDefaultSensor(sensorType);
        if (sensor != null) {
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST);
        } else {
            Log.d(getClass().getSimpleName(), "Sensor of type " + sensorType + " not found");
        }
    }

    private void unregisterListeners() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (Float.isNaN(event.values[0])) {
            unregisterListeners();
            registerListeners();
            return;
        }

        if (!started) {
            return;
        }

        float timestamp;
        if (referenceTimestamp == 0) {
            referenceTimestamp = event.timestamp;
            timestamp = 0f;
        } else {
            // convert event timestamp from ns to s (factor 10^9) with offset 0.05s to synchronize with audio
            timestamp = (event.timestamp - referenceTimestamp) / 1000000000f + 0.05f;
        }
        // adding the values to the corresponding list
        switch (event.sensor.getType()) {
            case Sensor.TYPE_GYROSCOPE:
                gyroData.add(new IMUEntry(timestamp, event.values.clone()));
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                linAccelData.add(new IMUEntry(timestamp, event.values.clone()));
                break;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void createEntryLists() {
        this.gyroData = new ArrayList<>((int)(1.5 * Settings.Sensor.Gyroscope.SAMPLES_PER_WINDOW));
        this.linAccelData = new ArrayList<>((int)(1.5 * Settings.Sensor.LinearAcceleration.SAMPLES_PER_WINDOW));
    }
}
