# ZED-2i-object-detection
Python scripts for object detection using the ZED stereo camera, featuring real-time depth estimation, bounding box tracking, and integration with OpenCV and deep learning models for robust spatial analysis.

detector_edit3.py -> printing x,y coordinates for the 'yolov8m' model
detector_edit5.py -> Outputs: Tracking ID, 3D position, Velocity, 3D dimensions, bounding box 2D & 3D
detector_edit9.py -> Depth to the relevant object (z co-ordinate)
detector_edit13.py -> Speed is 0.001s and 
test6.py ->
test7.py ->
test8.py -> 

3D position - Provides the 3D position of the object according to the camera as a 3D vector (x,y,z).
Velocity - Provides the velocity of the object in space as a 3D vector (x,y,z).
Dimensions - Provides the width, height and length of the object.(w,h,l)
2D bounding box - Defines the box surrounding the object in the image represented as four 2D points. [Four pixel coordinates]
3D bounding box - Defines the box surrounding the object in space represented as eight 3D points. [Eight 3D coordinates]
