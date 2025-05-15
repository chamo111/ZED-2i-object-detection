# ZED-2i-object-detection
Python scripts for object detection using the ZED stereo camera, featuring real-time depth estimation, bounding box tracking, and integration with OpenCV and deep learning models for robust spatial analysis.

detector_edit3.py -> printing x,y coordinates for the 'yolov8m' model <br>
detector_edit5.py -> Outputs: Tracking ID, 3D position, Velocity, 3D dimensions, bounding box 2D & 3D <br>
detector_edit9.py -> Depth to the relevant object (z co-ordinate) <br>
detector_edit13.py -> Speed is 0.001s and <br>
test6.py -> Detecting the mask (you can replace the ' .pt' file with your own pt file. <br>
test7.py -> Co-ordinates of the detected object's bounding box <br>
test8.py -> 3D position co-ordinates <br>

3D position - Provides the 3D position of the object according to the camera as a 3D vector (x,y,z). <br>
Velocity - Provides the velocity of the object in space as a 3D vector (x,y,z). <br>
Dimensions - Provides the width, height and length of the object.(w,h,l) <br>
2D bounding box - Defines the box surrounding the object in the image represented as four 2D points. [Four pixel coordinates] <br>
3D bounding box - Defines the box surrounding the object in space represented as eight 3D points. [Eight 3D coordinates] <br>
