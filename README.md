# StarFitter
Calibrates cameras by fitting star positions.

The following parameters are calculated:
- Azimuth, altitide and roll (rotation around vector defined by azimuth and altitide)
- Field of view
- Some n - 2 correction coefficients for radial lens corrections with an n-th order polynomial

Required inputs:
- one or more images with date and time encoded in the filename
- initial guesses of the parameters
  - optionally a set of manually matched stars to improve intial guesses

See the python files for usage of this tool.
