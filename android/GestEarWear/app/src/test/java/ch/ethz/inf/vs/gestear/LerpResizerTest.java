package ch.ethz.inf.vs.gestear;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import ch.ethz.inf.vs.gestear.datacollection.imu.IMUEntry;
import ch.ethz.inf.vs.gestear.datacollection.imu.LerpResizer;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class LerpResizerTest {

    private void compare(float[][][] expectedResult, float[][][] actualResult) {
        for (int i = 0; i < expectedResult.length; i++) {
            for (int j = 0; j < expectedResult[0].length; j++) {
                for (int k = 0; k < expectedResult.length; k++) {
                    assertTrue(Math.abs(expectedResult[i][j][k] - actualResult[i][j][k]) < 0.00001);
                }
            }
        }
    }

    @Test
    public void simpleTest() {
        List<IMUEntry> sequence = new ArrayList<>();
        float[] values = {1, 1, 1};
        for (int i = 0; i <= 10; i++) {
            sequence.add(new IMUEntry(i, values));
        }

        double startTimestamp = 0;
        float[][][] data = LerpResizer.getInstance().resizeSequence(sequence, 1, startTimestamp, 10, 2);

        float[][][] expectedResult = new float[2][5][3];
        for (int i = 0; i < expectedResult.length; i++) {
            for (int j = 0; j < expectedResult[0].length; j++) {
                for (int k = 0; k < expectedResult[0][0].length; k++) {
                    expectedResult[i][j][k] = 1f;
                }
            }
        }

        // test
        compare(expectedResult, data);
    }

    @Test
    public void increasingTest() {
        List<IMUEntry> sequence = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            float[] values = {i, i, i};
            sequence.add(new IMUEntry(i, values));
        }

        double startTimestamp = 0;
        float[][][] data = LerpResizer.getInstance().resizeSequence(sequence, 1, startTimestamp, 10, 2);

        float[][][] expectedResult = new float[2][5][3];
        for (int i = 0; i < expectedResult.length; i++) {
            for (int j = 0; j < expectedResult[0].length; j++) {
                for (int k = 0; k < expectedResult[0][0].length; k++) {
                    expectedResult[i][j][k] = i * expectedResult[0].length + j;
                }
            }
        }

        // test
        compare(expectedResult, data);
    }

    @Test
    public void interpolationTest() {
        List<IMUEntry> sequence = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            float[] values = {i + 0.5f, i + 0.5f, i + 0.5f};
            sequence.add(new IMUEntry(i + 0.5, values));
        }

        double startTimestamp = 0;
        float[][][] data = LerpResizer.getInstance().resizeSequence(sequence, 1, startTimestamp, 10, 2);

        float[][][] expectedResult = new float[2][5][3];
        for (int i = 0; i < expectedResult.length; i++) {
            for (int j = 0; j < expectedResult[0].length; j++) {
                for (int k = 0; k < expectedResult[0][0].length; k++) {
                    expectedResult[i][j][k] = i * expectedResult[0].length + j;
                }
            }
        }
        float[] startValues = {0.5f, 0.5f, 0.5f};
        expectedResult[0][0] = startValues;

        // test
        compare(expectedResult, data);
    }

    @Test
    public void increasingTest2() {
        List<IMUEntry> sequence = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            float[] values = {i + 0.5f, i + 0.5f, i + 0.5f};
            sequence.add(new IMUEntry(i * 0.5, values));
        }

        double startTimestamp = 0;
        float[][][] data = LerpResizer.getInstance().resizeSequence(sequence, 2, startTimestamp, 10, 2);

        float[][][] expectedResult = new float[2][10][3];
        int counter = 0;
        for (int i = 0; i < expectedResult.length; i++) {
            for (int j = 0; j < expectedResult[0].length; j++) {
                for (int k = 0; k < expectedResult[0][0].length; k++) {
                    expectedResult[i][j][k] = counter + 0.5f;
                }
                counter++;
            }
        }

        // test
        compare(expectedResult, data);
    }

    @Test
    public void interpolationTest2() {
        List<IMUEntry> sequence = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            float[] values = {i + 0.5f, i + 0.5f, i + 0.5f};
            sequence.add(new IMUEntry(i * 0.5 + 0.25, values));
        }

        double startTimestamp = 0;
        float[][][] data = LerpResizer.getInstance().resizeSequence(sequence, 2, startTimestamp, 10, 2);

        float[][][] expectedResult = new float[2][10][3];
        int counter = 0;
        for (int i = 0; i < expectedResult.length; i++) {
            for (int j = 0; j < expectedResult[0].length; j++) {
                for (int k = 0; k < expectedResult[0][0].length; k++) {
                    expectedResult[i][j][k] = counter;
                }
                counter++;
            }
        }
        float[] startValues = {0.5f, 0.5f, 0.5f};
        expectedResult[0][0] = startValues;

        // test
        compare(expectedResult, data);
    }
}