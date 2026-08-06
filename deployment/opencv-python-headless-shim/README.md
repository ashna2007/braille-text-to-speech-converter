# Headless OpenCV dependency shim

`inference-sdk` and Ultralytics declare a dependency on the desktop
`opencv-python` distribution even though this application does not use OpenCV
GUI features. Streamlit Community Cloud cannot provide the GLib libraries
required by that wheel.

This metadata-only local package uses the same distribution name and version
expected by those dependencies, but installs `opencv-python-headless` as the
actual provider of the `cv2` module. It contains no Python modules itself.
