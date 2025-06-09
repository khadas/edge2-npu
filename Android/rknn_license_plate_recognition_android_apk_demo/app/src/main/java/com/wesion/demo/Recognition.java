package com.wesion.demo;

public class Recognition {

    static {
        System.loadLibrary("demo");
    }

    public static native int native_init(String lpd_model, String lpd_label, String lpr_model, String lpr_label, String lpc_model, String lpc_label);
    public static native int native_deInit();
    public static native int native_identify(int width, int height, int channel, int flip, byte[] data, int[] ids, float[] scores, int[] boxes, byte[] points);

}
