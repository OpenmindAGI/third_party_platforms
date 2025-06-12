import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import websockets
import json
import queue
import os 
import sys
import asyncio

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm_armsdk import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller
from teleop.utils.episode_writer import EpisodeWriter

class WebSocketReceiver:
    """WebSocket client that receives pose data and puts it in a queue"""
    
    def __init__(self, uri="ws://localhost:8765", queue_size=10):
        self.uri = uri
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None
        self.loop = None
        
    def start(self):
        """Start the WebSocket receiver thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the WebSocket receiver thread"""
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._receive_data())
        
    async def _receive_data(self):
        """Async function to receive data from WebSocket"""
        while self.running:
            try:
                async with websockets.connect(self.uri) as websocket:
                    print(f"Connected to WebSocket server at {self.uri}")
                    
                    while self.running:
                        try:
                            # Receive data with timeout
                            message = await asyncio.wait_for(
                                websocket.recv(), 
                                timeout=1.0
                            )
                            
                            # Parse the received data
                            data = json.loads(message)
                            
                            # Convert lists back to numpy arrays
                            head_rmat = np.array(data['head_rmat']).reshape(3, 3)
                            left_wrist = np.array(data['left_wrist']).reshape(4, 4)
                            right_wrist = np.array(data['right_wrist']).reshape(4, 4)
                            left_hand = np.array(data['left_hand']).reshape(25, 3)
                            right_hand = np.array(data['right_hand']).reshape(25, 3)
                            
                            # Put data in queue (overwrites old data if queue is full)
                            try:
                                if self.data_queue.full():
                                    self.data_queue.get_nowait()  # Remove oldest
                                self.data_queue.put_nowait((
                                    head_rmat, left_wrist, right_wrist, 
                                    left_hand, right_hand
                                ))
                            except queue.Full:
                                pass  # Skip if queue is still full
                                
                        except asyncio.TimeoutError:
                            # No data received, continue
                            continue
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON: {e}")
                            continue
                            
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed, attempting to reconnect...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"WebSocket error: {e}")
                await asyncio.sleep(1)
                
    def get_latest_data(self, timeout=0.1):
        """Get the latest data from the queue
        
        Returns:
            tuple: (head_rmat, left_wrist, right_wrist, left_hand, right_hand)
            or None if no data is available
        """
        try:
            # Get the most recent data, discarding older entries
            data = None
            while True:
                data = self.data_queue.get(timeout=timeout)
                if self.data_queue.empty():
                    break
            return data
        except queue.Empty:
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')
    parser.add_argument('--websocket_uri', type = str, default = 'ws://192.168.1.162:8765', help = 'WebSocket server URI')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--hand', type=str, choices=['dex3', 'gripper', 'inspire1'], help='Select hand controller')

    args = parser.parse_args()
    print(f"args:{args}\n")

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    # tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name)
    ws_receiver = WebSocketReceiver(uri=args.websocket_uri)

    # arm
    if args.arm == 'G1_29':
        arm_ctrl = G1_29_ArmController()
        arm_ik = G1_29_ArmIK()
    elif args.arm == 'G1_23':
        arm_ctrl = G1_23_ArmController()
        arm_ik = G1_23_ArmIK()
    elif args.arm == 'H1_2':
        arm_ctrl = H1_2_ArmController()
        arm_ik = H1_2_ArmIK()
    elif args.arm == 'H1':
        arm_ctrl = H1_ArmController()
        arm_ik = H1_ArmIK()

    # hand
    if args.hand == "dex3":
        left_hand_array = Array('d', 75, lock = True)         # [input]
        right_hand_array = Array('d', 75, lock = True)        # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.hand == "gripper":
        left_hand_array = Array('d', 75, lock=True)
        right_hand_array = Array('d', 75, lock=True)
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
    elif args.hand == "inspire1":
        left_hand_array = Array('d', 75, lock = True)          # [input]
        right_hand_array = Array('d', 75, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    else:
        pass
    
    if args.record:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        recording = False
        
    try:
        # user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        # if user_input.lower() == 'r':
        time.sleep(5)
        ws_receiver.start()
        print("WebSocket receiver started, waiting for data...")

        # Wait for initial data
        wait_time = 0
        while wait_time < 5.0:  # Wait up to 5 seconds for initial data
            data = ws_receiver.get_latest_data(timeout=0.1)
            if data:
                print("Received initial data from WebSocket")
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = data
                break
            time.sleep(0.1)
            wait_time += 0.1
        else:
            print("Warning: No initial data received, using default values")

        arm_ctrl.speed_gradual_max()

        running = True
        while running:
            start_time = time.time()
            # head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()

            # Get latest data from WebSocket queue
            latest_data = ws_receiver.get_latest_data(timeout=0.001)  # Non-blocking

            if latest_data:
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = latest_data

            # send hand skeleton data to hand_ctrl.control_process
            if args.hand:
                left_hand_array[:] = left_hand.flatten()
                right_hand_array[:] = right_hand.flatten()

            # get current state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            # print(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

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
                # dex hand or gripper
                if args.hand == "dex3":
                    with dual_hand_data_lock:
                        left_hand_state = dual_hand_state_array[:7]
                        right_hand_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                elif args.hand == "gripper":
                    with dual_gripper_data_lock:
                        left_hand_state = [dual_gripper_state_array[1]]
                        right_hand_state = [dual_gripper_state_array[0]]
                        left_hand_action = [dual_gripper_action_array[1]]
                        right_hand_action = [dual_gripper_action_array[0]]
                elif args.hand == "inspire1":
                    with dual_hand_data_lock:
                        left_hand_state = dual_hand_state_array[:6]
                        right_hand_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                else:
                    print("No dexterous hand set.")
                    pass

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]

                if recording:
                    colors = {}
                    depths = {}
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_hand": {                                                                    
                            "qpos":   left_hand_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_hand": {                                                                    
                            "qpos":   right_hand_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": None, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_hand": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_hand": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": None, 
                    }
                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
            time.sleep(sleep_time)
            # print(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        if args.record:
            recorder.close()
        print("Finally, exiting program...")
        exit(0)