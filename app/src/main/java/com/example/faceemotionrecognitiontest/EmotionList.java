package com.example.faceemotionrecognitiontest;

import java.util.ArrayList;

public class EmotionList<T> extends ArrayList<float[]> {
    private final int limit;
    private int emotions;
    public EmotionList(int limit, int emotions) {
        this.limit = limit;
        this.emotions = emotions;
    }

    public void setEmotions(int emotions) {
        this.emotions = emotions;
        super.clear();
    }

    public boolean add(float[] array) {
        while (this.size() >= limit) {
            this.remove(0);
        }
        return super.add(array);
    }

    public float[] getEmotionAverages() {
        if (this.size() == 0) {
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

    public void removeLast() {
        if (this.size() != 0) {
            this.remove(0);
        }
    }

    public float[] getTail() {
        if (this.size() != 0) {
            return this.get(this.size() - 1);
        }
        else {
            return new float[] {};
        }
    }
}
