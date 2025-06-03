"""bag-to-images: 
a pyrealsense2 based document converter. .bag -> mp4 """

import os
import pyrealsense2 as rs
import numpy as np
import cv2

# === CONFIG ===
bag_file = r"C:/Users/orife/Downloads/stairs.bag"
output_dir = r"C:/Users/orife/Desktop/realsense_converter/test"
os.makedirs(output_dir, exist_ok=True)

# === SETUP PIPELINE ===
pipeline = rs.pipeline()
config = rs.config()

# Enable device from file
config.enable_device_from_file(bag_file, repeat_playback=False)

# Try to determine available streams first
try:
    # Start with a basic config to see what's available
    profile = pipeline.start(config)
    
    # Get the device from the pipeline
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  # Process as fast as possible
    
    print("Successfully started pipeline")
    
    i = 0
    try:
        while True:
            # Wait for frames with timeout
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            
            # Get color and depth frames if available
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            # Process color frame
            if color_frame and (i%24==0): #FRAME CONTROL 
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imwrite(os.path.join(output_dir, f"color_{i:04d}.png"), color_image)
                print(f"Saved color frame {i}")
            
            # # Process depth frame
            # if depth_frame:
            #     depth_image = np.asanyarray(depth_frame.get_data())
            #     # Convert depth to 8-bit for better visualization (optional)
            #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #     cv2.imwrite(os.path.join(output_dir, f"depth_{i:04d}.png"), depth_image)
            #     cv2.imwrite(os.path.join(output_dir, f"depth_colormap_{i:04d}.png"), depth_colormap)
            #     print(f"Saved depth frame {i}")
            
            if color_frame or depth_frame:
                i += 1
                
    except RuntimeError as e:
        print("End of bag file reached or error:", e)
    
    finally:
        pipeline.stop()

except RuntimeError as e:
    print(f"Failed to start pipeline: {e}")
    
    # Alternative approach: Try without specifying stream formats
    print("Trying alternative approach...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)
        
        i = 0
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                
                # Try to get any available frames
                for frame in frames:
                    if frame.is_color_frame() and (i %24==0):
                        color_image = np.asanyarray(frame.get_data())
                        cv2.imwrite(os.path.join(output_dir, f"color_{i:04d}.png"), color_image)
                        print(f"Saved color frame {i}")
                    # elif frame.is_depth_frame():
                    #     depth_image = np.asanyarray(frame.get_data())
                    #     cv2.imwrite(os.path.join(output_dir, f"depth_{i:04d}.png"), depth_image)
                    #     print(f"Saved depth frame {i}")
                
                i += 1
                
            except RuntimeError:
                print("End of bag file reached")
                break
                
        pipeline.stop()
        
    except Exception as e:
        print(f"Alternative approach also failed: {e}")
        
        # Final fallback: inspect bag file contents
        print("\nInspecting bag file contents...")
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device_from_file(bag_file)
            
            profile = pipeline.start(config)
            device = profile.get_device()
            
            print("Available streams in bag file:")
            for sensor in device.sensors:
                for stream_profile in sensor.stream_profiles:
                    print(f"  {stream_profile}")
                    
            pipeline.stop()
            
        except Exception as inspect_error:
            print(f"Could not inspect bag file: {inspect_error}")

print("Processing complete!")