package ch.ethz.inf.vs.gestear;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class PostProcessorTest {

    @Test
    public void simpleTest() {
        PostProcessor pp = new PostProcessor();
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(1) == PostProcessor.Gesture.SNAP_LEFT);
        assertTrue(pp.performedGesture(1) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(1) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(2) == PostProcessor.Gesture.SNAP_RIGHT);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(3) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.KNOCK_LEFT);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(0) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(3) == PostProcessor.Gesture.NULL);
        assertTrue(pp.performedGesture(6) == PostProcessor.Gesture.KNOCK_LEFT2X);
    }
}
