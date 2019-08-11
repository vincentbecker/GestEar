package ch.ethz.inf.vs.gestear;

import android.util.Log;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * A state machine for postprocessing
 * Gesture labels:
 * Null: 0
 * Snap left: 1
 * Snap right: 2
 * Knock left: 3
 * Knock right: 4
 * Clap: 5
 * Knock left 2x: 6
 * Knock right 2x: 7
 * Clap 2x: 8
 */
class PostProcessor2 {

    private static final String TAG = "POST_PROCESSOR2";
    private static final int N_CLASSES = 9;
    private static final int TEMPORAL_THRESHOLD = 5;
    private static final int WINDOW_LENGTH = 4;

    enum Gesture {NULL, SNAP_LEFT, SNAP_RIGHT, KNOCK_LEFT, KNOCK_RIGHT, CLAP, KNOCK_LEFT2X, KNOCK_RIGHT2X, CLAP2X}

    private int countSinceLastOutput = TEMPORAL_THRESHOLD;
    private Queue<RecognitionResult> resultQueue = new ConcurrentLinkedQueue<>();
    private RecognitionResult[] results = new RecognitionResult[WINDOW_LENGTH];
    private int[] counts = new int[N_CLASSES];
    private float[] weightsFFT = new float[N_CLASSES];
    private float[] weightsConfidence = new float[N_CLASSES];
    private int[] indexes = new int[N_CLASSES];

    Gesture performedGesture(RecognitionResult result) {
        resultQueue.add(result);
        countSinceLastOutput++;
        if (resultQueue.size() > WINDOW_LENGTH) {
            resultQueue.poll();
            if (countSinceLastOutput > TEMPORAL_THRESHOLD && resultQueue.peek().getLabel() != 0) {
                results = resultQueue.toArray(results);
                String s = "";
                for (RecognitionResult r : results) {
                    s += r.getLabel();
                }
                Log.d(TAG, "Labels: " + s);
                for (int i = 0; i < N_CLASSES; i++) {
                    counts[i] = 0;
                    weightsFFT[i] = 0;
                    weightsConfidence[i] = 0;
                    indexes[i] = -1;
                }
                for (int i = 0; i < WINDOW_LENGTH; i++) {
                    int label = results[i].getLabel();
                    counts[label]++;
                    weightsFFT[label] += results[i].getFftMagnitude();
                    weightsConfidence[label] += results[i].getProbability();
                    if (indexes[label] == -1 || label == 6 || label == 7 || label == 8) {
                        indexes[label] = i;
                    }
                }
                counts[0] = 0;
                weightsFFT[0] = 0;
                weightsConfidence[0] = 0;
                float winnerValue = 0;
                int winner = 0;
                float runnerupValue = 0;
                int runnerup = 0;
                for (int i = 0; i < N_CLASSES; i++) {
                    float value = weightsConfidence[i];
                    if (value > winnerValue) {
                        winnerValue = value;
                        winner = i;
                    }
                }
                for (int i = 0; i < N_CLASSES; i++) {
                    float value = weightsConfidence[i];
                    if (value > runnerupValue && i != winner) {
                        runnerupValue = value;
                        runnerup = i;
                    }
                }
                float winnerWeightsConfidenceAvg = weightsConfidence[winner] / counts[winner];
                // float winnerWeightsFFTAvg = weightsFFT[winner] / counts[winner];
                int output = winner;
                if ((winner == 3 || winner == 4 || winner == 5) && indexes[winner + 3] > indexes[winner]) {
                    float winnerDoubleWeightsConfidenceAvg = weightsConfidence[winner + 3] / counts[winner + 3];
                    // float winnerDoubleWeightsFFTAvg = weightsFFT[winner + 3] / counts[winner + 3];
                    if (winnerDoubleWeightsConfidenceAvg >= winnerWeightsConfidenceAvg - 0.04) {
                        output = winner + 3;
                    }
                } else if (weightsFFT[runnerup] > weightsFFT[winner] + 0.2) {
                    float runnerupWeightAvg = weightsConfidence[runnerup] / counts[runnerup];
                    if(runnerupWeightAvg >= winnerWeightsConfidenceAvg - 0.05) {
                        output = runnerup;
                    }
                }
                countSinceLastOutput = 0;
                return Gesture.values()[output];
            }
        }
        return Gesture.NULL;
    }
}
