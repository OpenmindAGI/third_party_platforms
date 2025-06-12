import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import platform
import threading
import queue

class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        device_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        
        # Use appropriate backend based on OS
        system = platform.system()
        if system == "Darwin":  # macOS
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_AVFOUNDATION)
        elif system == "Linux":
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        else:  # Windows or others
            self.cap = cv2.VideoCapture(self.id)
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Test if the camera can read frames
        if not self._can_read_frame():
            print(f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames. Exiting...")
            self.release()

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port = 5555, Unit_Test = False):
        print(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])
        
        # Performance optimization settings
        self.jpeg_quality = config.get('jpeg_quality', 50)  # Lower quality = faster
        self.skip_frames = config.get('skip_frames', 0)  # Skip frames to reduce load
        self.downscale_factor = config.get('downscale_factor', 1.0)  # Downscale images
        self.use_threading = config.get('use_threading', True)  # Use threading
        self.buffer_size = config.get('buffer_size', 1)  # Frame buffer size
        
        self.port = port
        self.Unit_Test = Unit_Test
        self.frame_counter = 0

        # Initialize head cameras
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            for device_id in self.head_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                self.head_cameras.append(camera)
        else:
            print(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # Set ZeroMQ context and socket with optimizations
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # ZMQ optimizations
        self.socket.setsockopt(zmq.SNDHWM, 1)  # High water mark - drop old messages
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.socket.setsockopt(zmq.IMMEDIATE, 1)  # Don't queue messages for disconnected peers
        
        self.socket.bind(f"tcp://*:{self.port}")

        # Threading setup
        if self.use_threading:
            self.frame_queue = queue.Queue(maxsize=self.buffer_size)
            self.capture_thread_running = True

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                print(f"[Image Server] Head camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")

        print("[Image Server] Image server has started, waiting for client connections...")

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = deque()
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        self.frame_times.append(current_time)
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        if self.use_threading:
            self.capture_thread_running = False
        for cam in self.head_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def _capture_frames(self):
        """Capture frames in a separate thread"""
        while self.capture_thread_running:
            head_frames = []
            for cam in self.head_cameras:
                if self.head_camera_type == 'opencv':
                    color_image = cam.get_frame()
                    if color_image is None:
                        continue
                head_frames.append(color_image)
            
            if len(head_frames) == len(self.head_cameras):
                # Drop old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(head_frames)

    def send_process(self):
        try:
            # Start capture thread if enabled
            if self.use_threading:
                capture_thread = threading.Thread(target=self._capture_frames)
                capture_thread.daemon = True
                capture_thread.start()

            while True:
                # Frame skipping
                if self.skip_frames > 0:
                    self.frame_counter += 1
                    if self.frame_counter % (self.skip_frames + 1) != 0:
                        continue

                # Get frames
                if self.use_threading:
                    try:
                        head_frames = self.frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                else:
                    head_frames = []
                    for cam in self.head_cameras:
                        if self.head_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                print("[Image Server] Head camera frame read is error.")
                                break
                        head_frames.append(color_image)
                    
                    if len(head_frames) != len(self.head_cameras):
                        break

                # Concatenate frames
                head_color = cv2.hconcat(head_frames)
                
                # Downscale if needed
                if self.downscale_factor < 1.0:
                    new_width = int(head_color.shape[1] * self.downscale_factor)
                    new_height = int(head_color.shape[0] * self.downscale_factor)
                    head_color = cv2.resize(head_color, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                full_color = head_color

                # Encode with specified quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                ret, buffer = cv2.imencode('.jpg', full_color, encode_param)
                if not ret:
                    print("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                # Send without blocking
                try:
                    self.socket.send(message, zmq.NOBLOCK)
                except zmq.Again:
                    # Socket would block, skip this frame
                    pass

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()

if __name__ == "__main__":
    config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [1080, 1920],  # Original resolution
        'head_camera_id_numbers': [0],
        
        # Performance optimizations
        'jpeg_quality': 30,  # Reduce JPEG quality (1-100, lower = faster)
        'skip_frames': 2,  # Skip every other frame (0 = no skip, 1 = skip 1, etc.)
        'downscale_factor': 0.3,  # Downscale to 50% (1.0 = no downscale)
        'use_threading': True,  # Use separate thread for capture
        'buffer_size': 1,  # Minimal buffer to reduce latency
    }

    server = ImageServer(config, Unit_Test=False)
    server.send_process()