import cv2
import imutils
import multiprocessing as mp
import threading
import time
from datetime import datetime
import numpy as np
from typing import Tuple, List, Optional
import queue
import logging
import sys
import os

class Streamer:
    """Component that reads video frames and sends them to the Detector."""
    
    def __init__(self, video_path: str, output_queue: mp.Queue):
        self.video_path = video_path
        self.output_queue = output_queue
        self.cap = None
        
    def start(self):
        """Start the streamer process."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video file {self.video_path}")
                return
                
            # Get video properties for better logging
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Streamer: Started reading video frames (FPS: {fps:.2f}, Total: {total_frames})")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Streamer: End of video reached")
                    break
                
                # Send frame to detector (blocking to prevent memory overflow)
                try:
                    self.output_queue.put(('frame', frame))
                except Exception as e:
                    print(f"Streamer: Error sending frame - {e}")
                    break
                    
        except Exception as e:
            print(f"Streamer: Critical error - {e}")
        finally:
            # Signal end of stream
            try:
                self.output_queue.put(('end', None))
            except:
                pass
            if self.cap:
                self.cap.release()
            print("Streamer: Finished processing video")


class Detector:
    """Component that performs motion detection on frames."""
    
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, detection_threshold=25, min_contour_area=500):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detection_threshold = detection_threshold
        self.min_contour_area = min_contour_area
        self.prev_frame = None
        self.first_frame_processed = False
        
    def detect_motion(self, frame) -> List[Tuple[int, int, int, int]]:
        """Perform motion detection on the frame and return bounding boxes."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if not self.first_frame_processed:
            self.prev_frame = gray_frame
            self.first_frame_processed = True
            return []
        else:
            # Calculate frame difference
            diff = cv2.absdiff(gray_frame, self.prev_frame)
            thresh = cv2.threshold(diff, self.detection_threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            # Filter contours by area and create bounding boxes
            detections = []
            for c in cnts:
                if cv2.contourArea(c) > self.min_contour_area:
                    (x, y, w, h) = cv2.boundingRect(c)
                    detections.append((x, y, w, h))
            
            self.prev_frame = gray_frame
            return detections
    
    def start(self):
        """Start the detector process."""
        print("Detector: Started motion detection")
        
        try:
            while True:
                try:
                    msg_type, data = self.input_queue.get(timeout=1)
                    
                    if msg_type == 'frame':
                        frame = data
                        
                        try:
                            detections = self.detect_motion(frame)
                            # Send frame with detections to presenter
                            self.output_queue.put(('frame_with_detections', (frame, detections)))
                        except Exception as e:
                            print(f"Detector: Error processing frame - {e}")
                            # Send frame without detections to maintain pipeline
                            self.output_queue.put(('frame_with_detections', (frame, [])))
                            
                    elif msg_type == 'end':
                        # Forward end signal
                        self.output_queue.put(('end', None))
                        break
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Detector: Error in main loop - {e}")
                    break
                    
        except Exception as e:
            print(f"Detector: Critical error - {e}")
        finally:
            print("Detector: Finished processing")


class Presenter:
    """Component that displays the video with detections and timestamp."""
    
    def __init__(self, input_queue: mp.Queue, blur_intensity=15, temporal_history_size=4):
        self.input_queue = input_queue
        self.blur_intensity = blur_intensity
        # Temporal smoothing for consistent blurring
        self.previous_detections = []
        self.detection_history = []  # Keep last N frames of detections
        self.history_size = temporal_history_size
        
    def apply_selective_blur(self, frame, detections, kernel_size=15):
        """Apply blur only to detected motion areas.
        
        Args:
            frame: Input frame
            detections: List of bounding boxes (x, y, w, h) where motion was detected
            kernel_size: Size of Gaussian kernel for blurring
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create a copy to avoid modifying the original
        blurred_frame = frame.copy()
        
        # Apply blur only to detected motion areas
        for (x, y, w, h) in detections:
            # Extract the region of interest (ROI)
            roi = frame[y:y+h, x:x+w]
            
            # Apply blur to this specific region
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            
            # Put the blurred region back
            blurred_frame[y:y+h, x:x+w] = blurred_roi
            
        return blurred_frame
    
    def smooth_detections_temporally(self, current_detections):
        """Apply temporal smoothing to reduce flickering between frames."""
        # Add current detections to history
        self.detection_history.append(current_detections)
        
        # Keep only last N frames
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # If we have enough history, use temporal smoothing
        if len(self.detection_history) >= 2:
            # Combine detections from recent frames
            smoothed_detections = []
            for frame_detections in self.detection_history:
                smoothed_detections.extend(frame_detections)
            
            # Remove duplicates and merge overlapping regions
            smoothed_detections = self.merge_overlapping_detections(smoothed_detections)
            return smoothed_detections
        
        # Not enough history yet, use current detections
        return current_detections
    
    def merge_overlapping_detections(self, detections):
        """Merge overlapping bounding boxes to reduce flickering."""
        if not detections:
            return []
        
        # Convert to list of tuples for easier processing
        boxes = list(detections)
        merged = []
        
        for box in boxes:
            x, y, w, h = box
            merged_any = False
            
            # Check if this box overlaps with any existing merged box
            for i, (mx, my, mw, mh) in enumerate(merged):
                # Calculate overlap
                overlap_x = max(0, min(x + w, mx + mw) - max(x, mx))
                overlap_y = max(0, min(y + h, my + mh) - max(y, my))
                overlap_area = overlap_x * overlap_y
                
                # If significant overlap, merge the boxes
                if overlap_area > 0:
                    # Expand the merged box to include both
                    new_x = min(x, mx)
                    new_y = min(y, my)
                    new_w = max(x + w, mx + mw) - new_x
                    new_h = max(y + h, my + mh) - new_y
                    
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_any = True
                    break
            
            # If no overlap found, add as new box
            if not merged_any:
                merged.append(box)
        
        return merged
        
    def draw_detections(self, frame, detections: List[Tuple[int, int, int, int]]):
        """Draw bounding boxes around detected motion areas."""
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    def draw_timestamp(self, frame):
        """Draw current timestamp on the frame."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    def start(self):
        """Start the presenter process."""
        print("Presenter: Started displaying video")
        displayed_frames = 0
        start_time = time.time()
        
        try:
            while True:
                try:
                    msg_type, data = self.input_queue.get(timeout=1)
                    
                    if msg_type == 'frame_with_detections':
                        frame, detections = data
                        displayed_frames += 1
                        
                        try:
                            # Apply temporal smoothing to reduce flickering
                            smoothed_detections = self.smooth_detections_temporally(detections)
                            
                            # Apply selective blur only to smoothed motion areas
                            blurred_frame = self.apply_selective_blur(frame, smoothed_detections, self.blur_intensity)
                            
                            # Draw detections on the frame (show original detections for debugging)
                            self.draw_detections(blurred_frame, detections)
                            
                            # Draw timestamp and FPS
                            self.draw_timestamp(blurred_frame)
                            self.draw_fps(blurred_frame, displayed_frames, start_time)
                            
                            # Display the frame with temporally smoothed blur
                            cv2.imshow('Motion Detection Pipeline', blurred_frame)
                            
                            # Break on 'q' key press
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("Presenter: User requested exit (Q key pressed)")
                                break
                                
                        except Exception as e:
                            print(f"Presenter: Error processing frame - {e}")
                            # Display original frame if processing fails
                            cv2.imshow('Motion Detection Pipeline', frame)
                            
                    elif msg_type == 'end':
                        print("Presenter: Received end signal - closing display")
                        break
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Presenter: Error in main loop - {e}")
                    break
                    
        except Exception as e:
            print(f"Presenter: Critical error - {e}")
        finally:
            cv2.destroyAllWindows()
            elapsed_time = time.time() - start_time
            fps = displayed_frames / elapsed_time if elapsed_time > 0 else 0
            print(f"Presenter: Finished displaying ({displayed_frames} frames, {fps:.2f} FPS)")
    
    def draw_fps(self, frame, frame_count, start_time):
        """Draw FPS counter on the frame."""
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


class PipelineSystem:
    """Main pipeline orchestrator that manages all three components."""
    
    def __init__(self, config):
        self.config = config
        self.video_path = config.video_path
        self.blur_intensity = config.blur_intensity
        self.streamer_to_detector = mp.Queue(maxsize=config.queue_size)
        self.detector_to_presenter = mp.Queue(maxsize=config.queue_size)
        
    def start(self):
        """Start the complete pipeline system."""
        print("Starting Pipeline System...")
        
        # Create components
        streamer = Streamer(self.video_path, self.streamer_to_detector)
        detector = Detector(self.streamer_to_detector, self.detector_to_presenter, 
                           self.config.detection_threshold, self.config.min_contour_area)
        presenter = Presenter(self.detector_to_presenter, self.blur_intensity, 
                          self.config.temporal_history_size)
        
        # Create processes
        streamer_process = mp.Process(target=streamer.start)
        detector_process = mp.Process(target=detector.start)
        presenter_process = mp.Process(target=presenter.start)
        
        # Start all processes
        streamer_process.start()
        detector_process.start()
        presenter_process.start()
        
        print("All processes started")
        
        # Wait for either presenter to finish (user closes window) or streamer to finish (video ends)
        def monitor_video_end():
            """Monitor when video ends and signal shutdown."""
            streamer_process.join()
            print("Video ended - initiating automatic system shutdown...")
            # Give a moment for the 'end' signal to propagate
            time.sleep(0.5)
            # Terminate presenter to trigger shutdown
            if presenter_process.is_alive():
                presenter_process.terminate()
        
        # Start monitoring thread for video end
        monitor_thread = threading.Thread(target=monitor_video_end)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for presenter to finish (either user closes window or video ends)
        presenter_process.join()
        
        # Terminate other processes
        streamer_process.terminate()
        detector_process.terminate()
        
        # Wait for processes to finish
        streamer_process.join()
        detector_process.join()
        
        print("Pipeline System finished")


class Config:
    """Configuration class for the pipeline system."""
    
    def __init__(self):
        self.video_path = "People - 6387.mp4"
        self.blur_intensity = 15  # 5: Light, 15: Medium, 25: Heavy
        self.queue_size = 30
        self.detection_threshold = 25
        self.min_contour_area = 500
        self.temporal_history_size = 5


def main():
    """Main function to run the pipeline system."""
    print("=" * 60)
    print("Motion Detection Pipeline System")
    print("=" * 60)
    
    config = Config()
    
    # Validate video file exists
    if not os.path.exists(config.video_path):
        print(f"Error: Video file '{config.video_path}' not found!")
        print("Please ensure the video file exists in the current directory.")
        return
    
    print(f"Video: {config.video_path}")
    print(f"Blur Intensity: {config.blur_intensity}")
    print(f"Queue Size: {config.queue_size}")
    print(f"Detection Threshold: {config.detection_threshold}")
    print(f"Min Contour Area: {config.min_contour_area}")
    print(f"Temporal History: {config.temporal_history_size} frames")
    print("-" * 60)
    
    try:
        # Create and start the pipeline
        pipeline = PipelineSystem(config)
        pipeline.start()
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        print("System shutdown complete")


if __name__ == "__main__":
    main()
