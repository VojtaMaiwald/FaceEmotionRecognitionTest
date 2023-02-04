package com.example.faceemotionrecognitiontest;

import java.util.ArrayList;

public class EmotionList<T> extends ArrayList<float[]> {
    private final int limit;
    private final int emotions;
    public EmotionList(int limit, int emotions) {
        this.limit = limit;
        this.emotions = emotions;
    }

    @Override
    public boolean add(float[] array) {
        while (super.size() >= limit) {
            super.remove(0);
        }
        return super.add(array);
    }

    public float[] getEmotionAverages() {
        if (super.size() == 0) {
            return null;
        }
        float[] averages = new float[emotions];
        for (int i = 0; i < emotions; i++) {
            averages[i] = 0;
        }

        for (float[] detection : this) {
            for (int i = 0; i < emotions; i++) {
                averages[i] += detection[i];
            }
        }

        for (int i = 0; i < emotions; i++) {
            averages[i] /= this.size();
        }

        return averages;
    }
}
