import cv2
import time
import logging
import argparse
import json
import os
from datetime import datetime
from collections import deque
import numpy as np
import serial
from serial.serialutil import SerialException


class MotorController:
    """Simple Arduino motor controller over serial. Sends single-byte commands 'L' or 'R'."""
    def __init__(self, port, baud=9600, logger=None):
        self.logger = logger
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            # give Arduino time to reset and initialize serial
            time.sleep(2)
            if self.logger:
                self.logger.info(f"Connected to Arduino on {port}")
        except SerialException as e:
            if self.logger:
                self.logger.error(f"Failed to connect to Arduino on {port}: {e}")
            self.ser = None

    def move(self, direction_char):
        """Send a single-character direction command ('L' or 'R')."""
        if not self.ser:
            return False
        try:
            if isinstance(direction_char, str):
                b = direction_char[0].encode()
            else:
                b = bytes([direction_char])
            self.ser.write(b)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to send serial command: {e}")
            return False

    def close(self):
        if getattr(self, 'ser', None):
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

class FaceDirectionTracker:
    def __init__(self, config_file="config.json"):
        """Initialize the Face Direction Tracker with configuration."""
        self.config = self.load_config(config_file)
        self.setup_logging()
        
        # Initialize tracking variables
        self.prev_center = None
        self.prev_time = time.time()
        self.direction = "Center"
        self.speed = 0
        self.face_history = deque(maxlen=self.config['smoothing_frames'])
        self.direction_history = deque(maxlen=10)
        
        # Initialize camera and face detector
        self.cap = None
        self.face_cascade = None
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            # Only horizontal directions + center are used for motor control
            'directions': {'Left': 0, 'Right': 0, 'Center': 0}
        }

        # Motor control state (initialized later in run)
        self.motor_controller = None
        self.last_motor_command_time = 0.0
        self.last_motor_direction_sent = None

    def load_config(self, config_file):
        """Load configuration from JSON file or create default."""
        default_config = {
            'camera_index': 0,
            'window_width': 640,
            'window_height': 480,
            'detection_scale_factor': 1.1,
            'detection_min_neighbors': 5,
            'movement_threshold': 8,
            'smoothing_frames': 6,
            'log_level': 'INFO',
            'save_video': False,
            'video_output': 'face_tracking_output.avi',
            'show_fps': True,
            'show_stats': True
        }
        # Motor control defaults
        default_config.update({
            'enable_motor': True,
            'arduino_port': 'COM10',
            'arduino_baud': 9600,
            'motor_command_interval': 0.5,
        })
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"[WARNING] Could not load config file: {e}. Using defaults.")
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"[INFO] Created default config file: {config_file}")
        
        return default_config

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['log_level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_camera(self):
        """Initialize camera with error handling."""
        try:
            self.cap = cv2.VideoCapture(self.config['camera_index'])
            if not self.cap.isOpened():
                raise Exception("Camera not accessible")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['window_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['window_height'])
            
            self.logger.info(f"Camera initialized successfully (index: {self.config['camera_index']})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False

    def initialize_face_detector(self):
        """Initialize face cascade classifier."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade classifier")
            
            self.logger.info("Face detector initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            return False

    def smooth_position(self, center):
        """Apply smoothing to face position."""
        self.face_history.append(center)
        if len(self.face_history) < 2:
            return center
        
        # Calculate weighted average
        weights = np.linspace(0.1, 1.0, len(self.face_history))
        weights /= weights.sum()
        
        avg_x = sum(pos[0] * w for pos, w in zip(self.face_history, weights))
        avg_y = sum(pos[1] * w for pos, w in zip(self.face_history, weights))
        
        return (int(avg_x), int(avg_y))

    def calculate_direction_and_speed(self, current_center):
        """Calculate movement direction and speed with improved logic."""
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 0.0001
        if self.prev_center:
            dx = current_center[0] - self.prev_center[0]
            # Only consider horizontal movement for direction
            if abs(dx) < self.config['movement_threshold']:
                direction = "Center"
            else:
                direction = "Right" if dx > 0 else "Left"

            # Speed still computed from euclidean distance for a reasonable magnitude
            dy = current_center[1] - self.prev_center[1]
            distance = (dx**2 + dy**2)**0.5
            speed = distance / dt if dt > 0 else 0.0

            # Update direction history for stability (use recent majority)
            self.direction_history.append(direction)
            if len(self.direction_history) >= 3:
                recent = list(self.direction_history)[-3:]
                direction = max(set(recent), key=recent.count)

            # Save results
            self.direction = direction
            self.speed = speed
            # Only increment stats for keys we track
            if direction in self.stats['directions']:
                self.stats['directions'][direction] += 1

        self.prev_center = current_center
        self.prev_time = current_time

    def draw_overlay(self, frame, faces):
        """Draw tracking information overlay on frame."""
        # Calculate FPS
        current_time = time.time()
        self.fps_counter.append(current_time)
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
        else:
            fps = 0

        # Draw face detection
        for (x, y, w, h) in faces:
            # Face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Face center
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            
            # Face ID
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Information panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw tracking information
        y_offset = 35
        cv2.putText(frame, f"Direction: {self.direction}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Speed: {self.speed:.1f} px/s", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.config['show_fps']:
            y_offset += 25
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        if self.config['show_stats']:
            y_offset += 25
            cv2.putText(frame, f"Faces: {len(faces)}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' for stats", (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def print_statistics(self):
        """Print tracking statistics."""
        print("\n" + "="*50)
        print("FACE TRACKING STATISTICS")
        print("="*50)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Faces detected: {self.stats['faces_detected']}")
        print(f"Detection rate: {(self.stats['faces_detected']/max(self.stats['total_frames'], 1)*100):.1f}%")
        print("\nDirection distribution:")
        for direction, count in self.stats['directions'].items():
            percentage = (count / max(sum(self.stats['directions'].values()), 1)) * 100
            print(f"  {direction}: {count} ({percentage:.1f}%)")
        print("="*50)

    def run(self):
        """Main tracking loop."""
        if not self.initialize_camera():
            return False
        
        if not self.initialize_face_detector():
            self.cap.release()
            return False

        self.logger.info("Starting Face Direction Tracker...")
        # Initialize motor controller if enabled
        if self.config.get('enable_motor'):
            try:
                self.motor_controller = MotorController(self.config.get('arduino_port'),
                                                        self.config.get('arduino_baud'),
                                                        logger=self.logger)
            except Exception as e:
                self.logger.error(f"Motor controller init failed: {e}")
                self.motor_controller = None
        
        # Video writer setup if saving video
        video_writer = None
        if self.config['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                self.config['video_output'], fourcc, 20.0, 
                (self.config['window_width'], self.config['window_height'])
            )

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break

                self.stats['total_frames'] += 1
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=self.config['detection_scale_factor'],
                    minNeighbors=self.config['detection_min_neighbors']
                )

                if len(faces) > 0:
                    self.stats['faces_detected'] += 1
                    
                    # Use the largest face for tracking
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    
                    # Calculate face center and apply smoothing
                    raw_center = (x + w // 2, y + h // 2)
                    smooth_center = self.smooth_position(raw_center)
                    
                    # Calculate direction and speed
                    self.calculate_direction_and_speed(smooth_center)

                    # Motor control: send L/R command on horizontal movement with simple debounce
                    try:
                        if self.config.get('enable_motor') and self.motor_controller:
                            now_t = time.time()
                            if self.direction in ('Left', 'Right'):
                                interval = self.config.get('motor_command_interval', 0.5)
                                if (now_t - self.last_motor_command_time) >= interval:
                                    cmd = 'L' if self.direction == 'Left' else 'R'
                                    sent = self.motor_controller.move(cmd)
                                    if sent:
                                        self.last_motor_command_time = now_t
                                        self.last_motor_direction_sent = self.direction
                                        self.logger.debug(f"Sent motor command: {cmd}")
                            else:
                                # Optionally, could send a center/stop command if Arduino supports it
                                pass
                    except Exception as me:
                        self.logger.error(f"Motor command error: {me}")

                # Draw overlay information
                self.draw_overlay(frame, faces)

                # Save video frame if enabled
                if video_writer:
                    video_writer.write(frame)

                # Display frame
                cv2.imshow("Enhanced Face Direction Tracker", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    self.print_statistics()

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            # Close motor serial if open
            try:
                if self.motor_controller:
                    self.motor_controller.close()
            except Exception:
                pass
            self.print_statistics()
            self.logger.info("Face Direction Tracker stopped")

        return True

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Face Direction Tracker")
    parser.add_argument("--config", default="config.json", 
                       help="Configuration file path (default: config.json)")
    parser.add_argument("--camera", type=int, 
                       help="Camera index to use (overrides config)")
    parser.add_argument("--save-video", action="store_true",
                       help="Save tracking video to file")
    
    args = parser.parse_args()
    
    # Create tracker instance
    tracker = FaceDirectionTracker(args.config)
    
    # Override config with command line arguments
    if args.camera is not None:
        tracker.config['camera_index'] = args.camera
    if args.save_video:
        tracker.config['save_video'] = True
    
    # Run the tracker
    success = tracker.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
