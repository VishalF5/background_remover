import cv2
import numpy as np
import rembg
import streamlit as st

# Ask user to upload an image
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
  # Read image from file
  image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
  image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)

  # Display image
  st.image(image, caption="Original image", use_column_width=True)

  while True:
    # Ask user to draw a rectangle over an object
    roi = cv2.selectROI("Original image", image)
    
    # Crop ROI and pass to rembg library
    roi_image = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    object_image = rembg.remove_bg(roi_image)

    # Draw outline around object
    contours, _ = cv2.findContours(object_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(object_image, contours, -1, (0, 255, 0), 3)

    # Overlay object outline on original image
    image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = object_image
    
    # Display output image
    st.image(image, caption="Output image", use_column_width=True)
    
    # Wait for user input
    key = st.button("Continue")
    if not key:
      break

