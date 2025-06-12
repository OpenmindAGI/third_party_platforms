import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import asyncio
import websockets
import json
import queue

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.image_server.image_client_fast import ImageClient
from teleop.utils.episode_writer import EpisodeWriter

class WebSocketServer:
    """WebSocket server that runs in a separate thread and sends TV wrapper data"""
    
    def __init__(self, host="0.0.0.0", port=8765, send_latest=True):
        self.host = host
        self.port = port
        self.data_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.connected_clients = set()
        self.send_latest = send_latest  # If True, always send latest data; if False, send sequentially
        
        # Statistics tracking
        self.stats_lock = threading.Lock()
        self.send_count = 0
        self.last_stats_time = time.time()
        self.last_stats_count = 0
        
    def start(self):
        """Start the WebSocket server in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"WebSocket server thread started")
        
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def send_data(self, head_rmat, left_wrist, right_wrist, left_hand, right_hand):
        """Add data to the queue to be sent to all connected clients"""
        data = {
            "head_rmat": head_rmat.tolist() if isinstance(head_rmat, np.ndarray) else head_rmat,
            "left_wrist": left_wrist.tolist() if isinstance(left_wrist, np.ndarray) else left_wrist,
            "right_wrist": right_wrist.tolist() if isinstance(right_wrist, np.ndarray) else right_wrist,
            "left_hand": left_hand.tolist() if isinstance(left_hand, np.ndarray) else left_hand,
            "right_hand": right_hand.tolist() if isinstance(right_hand, np.ndarray) else right_hand,
            "timestamp": time.time()
        }
        
        try:
            # If queue is full, remove the oldest item
            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put_nowait(data)
            # print(f"Sent data at {time.time()}")
        except queue.Full:
            pass

    def print_stats(self):
        """Print sending frequency statistics"""
        with self.stats_lock:
            current_time = time.time()
            time_diff = current_time - self.last_stats_time
            count_diff = self.send_count - self.last_stats_count
            
            if time_diff > 0:
                frequency = count_diff / time_diff
                queue_size = self.data_queue.qsize()
                
                print(f"\n--- WebSocket Stats ---")
                print(f"Send frequency: {frequency:.1f} Hz")
                print(f"Total sent: {self.send_count} messages")
                print(f"Connected clients: {len(self.connected_clients)}")
                print(f"Queue size: {queue_size}/{self.data_queue.maxsize}")
                print(f"Mode: {'Latest only' if self.send_latest else 'Sequential'}")
                print("----------------------\n")
                
                self.last_stats_time = current_time
                self.last_stats_count = self.send_count
            
    def _run_server(self):
        """Run the async server in a new event loop"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        start_server = websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        
        loop.run_until_complete(start_server)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        print(f"Clients should connect to: ws://192.168.1.162:{self.port}")
        
        # Run the broadcast task
        loop.create_task(self._broadcast_data())
        
        # Run forever
        loop.run_forever()
        
    async def _handle_client(self, websocket, path):
        """Handle a new client connection"""
        self.connected_clients.add(websocket)
        client_address = websocket.remote_address
        print(f"Client connected from {client_address}")
        
        try:
            # Keep the connection open
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)
            print(f"Client disconnected from {client_address}")
            
    async def _broadcast_data(self):
        """Continuously broadcast data to all connected clients"""
        while self.running:
            try:
                data = None
                
                if self.send_latest:
                    # Get the most recent data, discarding older entries
                    while True:
                        try:
                            data = self.data_queue.get_nowait()
                        except queue.Empty:
                            break
                else:
                    # Get data sequentially (FIFO)
                    try:
                        data = self.data_queue.get(timeout=0.1)
                    except queue.Empty:
                        pass
                
                if data and self.connected_clients:
                    # Send to all connected clients
                    message = json.dumps(data)
                    disconnected = set()
                    
                    for client in self.connected_clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                        except Exception as e:
                            print(f"Error sending to client: {e}")
                            disconnected.add(client)
                    
                    # Update statistics
                    with self.stats_lock:
                        self.send_count += len(self.connected_clients) - len(disconnected)
                    
                    # Remove disconnected clients
                    self.connected_clients -= disconnected
                    
                await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"Broadcast error: {e}")
                await asyncio.sleep(0.001)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--websocket_port', type = int, default = 8765, help = 'WebSocket server port')
    parser.add_argument('--send_latest', action = 'store_true', default = True, help = 'Send only latest data (low latency mode)')
    parser.add_argument('--send_sequential', dest = 'send_latest', action = 'store_false', help = 'Send all data sequentially')

    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--hand', type=str, choices=['dex3', 'gripper', 'inspire1'], help='Select hand controller')

    args = parser.parse_args()
    print(f"args:{args}\n")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        'fps': 30,
        'head_camera_type': 'opencv',
        'head_camera_image_shape': [216, 384],  # Head camera resolution
        'head_camera_id_numbers': [0],
    }

    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name)
    
    # Initialize WebSocket server
    ws_server = WebSocketServer(host="0.0.0.0", port=args.websocket_port)
    ws_server.start()
    
    if args.record:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        recording = False
        
    try:
        # user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        # if user_input.lower() == 'r':

        # Wait a bit for systems to initialize
        print("Waiting for systems to initialize...")
        time.sleep(5)
        
        print(f"WebSocket server broadcasting pose data on port {args.websocket_port}")
        
        if True:
            running = True
            print("Teleop started!")

            # Stats printing variables
            stats_interval = 5.0  # Print stats every 5 seconds
            last_stats_print = time.time()
            
            while running:
                start_time = time.time()
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()

                # print(head_rmat, left_wrist, right_wrist, left_hand, right_hand)
                # Send data via WebSocket to all connected clients
                ws_server.send_data(head_rmat, left_wrist, right_wrist, left_hand, right_hand)

                # Print statistics periodically
                current_time = time.time()
                if current_time - last_stats_print >= stats_interval:
                    ws_server.print_stats()
                    last_stats_print = current_time
                    
                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s') and args.record:
                    recording = not recording # state flipping
                    if recording:
                        if not recorder.create_episode():
                            recording = False
                    else:
                        recorder.save_episode()

                # record data
                if args.record:

                    # head image
                    current_tv_image = tv_img_array.copy()
                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()

                    if recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]

                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
                # print(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:
        # Stop WebSocket server
        ws_server.stop()

        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        print("Finally, exiting program...")
        exit(0)