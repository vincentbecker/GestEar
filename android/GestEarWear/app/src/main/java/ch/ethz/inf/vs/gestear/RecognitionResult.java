package ch.ethz.inf.vs.gestear;

class RecognitionResult {

    private int label;
    private float probability;
    private float fftMagnitude;

    int getLabel() {
        return label;
    }

    float getProbability() {
        return probability;
    }

    float getFftMagnitude() {
        return fftMagnitude;
    }

    void setFftMagnitude(float fftMagnitude) {
        this.fftMagnitude = fftMagnitude;
    }

    RecognitionResult(int label, float probability) {
        this.label = label;
        this.probability = probability;
    }
}
