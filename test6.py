from ultralytics import YOLO
import pyzed.sl as sl
import numpy as np
import cv2

# Load the YOLO model
model = YOLO("/usr/local/zed/samples/object detection/custom detector/python/pytorch_yolov8/bestnew.pt")
model.to('cuda')  # Move the model to GPU

# Initialize the ZED camera
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD2K resolution
init_params.camera_fps = 60  # Set fps to 60
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Set depth mode to performance
init_params.coordinate_units = sl.UNIT.METER  # Use meter units for depth measurements
init_params.depth_maximum_distance = 10  # Set maximum depth distance to 10 meters
init_params.enable_image_enhancement = True  # Enable image enhancement
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE  # Use image coordinate system

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(repr(err))
    exit(-1)

# Set runtime parameters
runtime_param = sl.RuntimeParameters()
runtime_param.enable_depth = True
runtime_param.enable_fill_mode = True
runtime_param.confidence_threshold = 80
runtime_param.texture_confidence_threshold = 100
runtime_param.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
runtime_param.remove_saturated_areas = True

image = sl.Mat()
depth = sl.Mat()

while True:
    # Grab an image
    if zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # Convert the image to BGR format (correct color representation)
        color_image = np.asanyarray(image.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)  # Correct color format
        depth_image = np.asanyarray(depth.get_data())

        # Perform object detection with YOLO
        results = model.predict(source=color_image, stream=True)

        # Process the results
        for result in results:
            # Draw the masks in red
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # Convert masks to numpy array
                for mask in masks:
                    # Ensure the mask is in the correct shape and type
                    mask = mask.astype(np.uint8)

                    # Resize the mask to match the image size
                    mask_resized = cv2.resize(mask, (color_image.shape[1], color_image.shape[0]))

                    # Create an overlay where the mask is applied
                    color_mask = np.zeros_like(color_image)
                    color_mask[:, :, 2] = mask_resized * 255  # Color the mask in red

                    # Apply the mask to the image with some transparency
                    color_image = cv2.addWeighted(color_image, 1, color_mask, 0.5, 0)

        # Display the color and depth images
        cv2.imshow("Color View", color_image)
        cv2.imshow("Depth View", depth_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

# Close the camera
zed.close()

