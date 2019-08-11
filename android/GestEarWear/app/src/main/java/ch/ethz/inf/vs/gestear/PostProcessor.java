package ch.ethz.inf.vs.gestear;

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
class PostProcessor {

    private static final String TAG = "POST_PROCESSOR";
    private static final int TEMPORAL_THRESHOLD = 5;

    enum Gesture {NULL, SNAP_LEFT, SNAP_RIGHT, KNOCK_LEFT, KNOCK_RIGHT, CLAP, KNOCK_LEFT2X, KNOCK_RIGHT2X, CLAP2X}

    private Gesture lastGesture = Gesture.NULL;
    private int countSinceLastOutput = TEMPORAL_THRESHOLD + 1;
    private int waitingCount = 0;

    Gesture performedGesture(int gestureIndex) {
        countSinceLastOutput++;
        Gesture inputGesture = Gesture.values()[gestureIndex];
        if (countSinceLastOutput > TEMPORAL_THRESHOLD) {
            if (inputGesture == Gesture.NULL) {
                waitingCount++;
                if (waitingCount > 2) {
                    return returnLastGesture();
                }
                return Gesture.NULL;
            } else if (inputGesture == Gesture.KNOCK_LEFT || inputGesture == Gesture.KNOCK_RIGHT || inputGesture == Gesture.CLAP) {
                waitingCount++;
                if (lastGesture == Gesture.NULL) {
                    waitingCount = 1;
                    lastGesture = inputGesture;
                } else if (waitingCount > 2) {
                    return returnLastGesture();
                }
                return Gesture.NULL;
            } else {
                countSinceLastOutput = 0;
                lastGesture = Gesture.NULL;
                return inputGesture;
            }
        }
        return Gesture.NULL;
    }

    private Gesture returnLastGesture() {
        if (lastGesture != Gesture.NULL) {
            Gesture output = lastGesture;
            lastGesture = Gesture.NULL;
            waitingCount = 0;
            return output;
        }
        return Gesture.NULL;
    }
}
