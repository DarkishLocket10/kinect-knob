// Color-camera exposure control bridge for the Kinect v2.
//
// libfreenect2 v0.2 has setColorAutoExposure / setColorSemiAutoExposure /
// setColorManualExposure on Freenect2Device, but the pinned python binding
// (freenect2 0.2.3) never wrapped them. Its Freenect2DeviceRef is just the
// raw Freenect2Device* cast to void* (see freenect2-c.cpp in the binding
// sources), so this shim takes that same pointer back and calls the C++ API.
// Built in the Docker image into /usr/local/lib/libkk_exposure.so and loaded
// from python with ctypes (src/kinectknob/capture/kv2_exposure.py).
//
// Capping integration time is the only real motion-blur lever the camera
// has: in dim rooms auto-exposure stretches toward ~33 ms (and halves the
// color stream to 15 fps), smearing fast hands beyond what the landmark
// model can track.

#include <libfreenect2/libfreenect2.hpp>

extern "C" {

int kk_set_color_auto_exposure(void *device, float exposure_compensation) {
    if (device == nullptr) return -1;
    static_cast<libfreenect2::Freenect2Device *>(device)
        ->setColorAutoExposure(exposure_compensation);
    return 0;
}

int kk_set_color_semi_auto_exposure(void *device, float pseudo_exposure_time_ms) {
    if (device == nullptr) return -1;
    static_cast<libfreenect2::Freenect2Device *>(device)
        ->setColorSemiAutoExposure(pseudo_exposure_time_ms);
    return 0;
}

int kk_set_color_manual_exposure(void *device, float integration_time_ms,
                                 float analog_gain) {
    if (device == nullptr) return -1;
    static_cast<libfreenect2::Freenect2Device *>(device)
        ->setColorManualExposure(integration_time_ms, analog_gain);
    return 0;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Indicator LED control (the glowing white Xbox logo on the front).
//
// libfreenect2 exposes Freenect2Device::setLedStatus(LedSettings) but the
// python binding never wrapped it either. The logo is purely an indicator —
// the depth IR illuminators are driven by the depth firmware and are not
// touched by this command. LedId is 0 or 1 (the logo is 0; 1 is set too for
// completeness), level is 0..1000.

#include <libfreenect2/led_settings.h>

extern "C" {

int kk_set_led(void *device, int led_id, int level) {
    if (device == nullptr) return -1;
    if (led_id < 0 || led_id > 1 || level < 0 || level > 1000) return -2;
    libfreenect2::LedSettings s;
    s.LedId = static_cast<uint16_t>(led_id);
    s.Mode = 0;  // constant
    s.StartLevel = static_cast<uint16_t>(level);
    s.StopLevel = static_cast<uint16_t>(level);
    s.IntervalInMs = 0;
    s.Reserved = 0;
    static_cast<libfreenect2::Freenect2Device *>(device)->setLedStatus(s);
    return 0;
}

}  // extern "C"
