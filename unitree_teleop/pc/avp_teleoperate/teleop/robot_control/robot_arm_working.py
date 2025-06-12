import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

import asyncio
import json
import time
import websockets
from typing import Optional, Dict, Any

G1_23_Num_Motors = 35

class MotorState:
    def __init__(self):
        self.q = 0.0
        self.dq = 0.0

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_23_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26

class G1_23_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRollNotUsed = 13
    kWaistPitchNotUsed = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristyawNotUsed = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class G1_23_ArmController_WS:
    def __init__(self, server_url: str = "ws://192.168.1.124:8765"):
        print("Initialize G1_23_ArmController...")
        self.server_url = server_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = True
        self.message_counter = 0
        # Controller parameters
        self.q_target = np.zeros(10)
        self.tauff_target = np.zeros(10)
        self.ready_to_send = False  

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        self.pending_sends = 0

        # Initialize buffers and messages
        self.lowstate_buffer = DataBuffer()
        self.msg_dict = {
            'target_tau': [0.0] * G1_23_Num_Motors,
            'target_q': [0.0] * G1_23_Num_Motors
        }

        # Statistics
        self.stats = {
            'lowstate_received': 0,
            'lowcmd_sent': 0,
            'arm_sdk_sent': 0,
            'connection_time': 0
        }

        self.ctrl_lock = threading.Lock()

        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop)
        self.async_thread.daemon = True
        self.async_thread.start()

        # Wait for connection and initial data
        print("[G1_23_ArmController] Waiting for WebSocket connection...")
        while not self.websocket or not self.websocket.open:
            time.sleep(0.1)
        
        print("[G1_23_ArmController] Waiting for initial lowstate data...")
        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
        
        print("WebSocket connected and receiving data!")

        # Initialize motor states
        self.all_motor_q = self.get_current_motor_q()
        print(f"Current all body motor state q:\n{self.all_motor_q}")
        print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}")

        # Start control thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()

        print("Initialize G1_23_ArmController OK!\n")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_main())

    async def _async_main(self):
        """Main async function"""
        try:
            await self.connect()
            await self.receive_messages()
        except Exception as e:
            print(f"Error in async main: {e}")
        finally:
            await self.disconnect()

    async def connect(self):
        """Connect to the WebSocket server"""
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        print("Connected!")
        self.stats['connection_time'] = time.time()

    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def send_arm_sdk_async(self, msg: dict):
        """Async send with better error tracking"""
        if not self.websocket:
            print("ERROR: Not connected!")
            return False
            
        try:
            message = {
                'type': 'lowcmd',
                'data': msg,
                'timestamp': time.time(),
                'id': self.stats['arm_sdk_sent']
            }
            
            print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Attempting to send message {self.stats['arm_sdk_sent']}")
            
            # Check websocket state
            if self.websocket.closed:
                print(f"ERROR: WebSocket is closed! State: {self.websocket.state}")
                return False
            
            # Try to send
            await self.websocket.send(json.dumps(message))
            
            print(f"[{time.strftime('%H:%M:%S.%f')[:-3]}] Successfully sent message {self.stats['arm_sdk_sent']}")
            self.stats['arm_sdk_sent'] += 1
            return True
            
        except websockets.exceptions.ConnectionClosed as e:
            print(f"ERROR: Connection closed during send {self.stats['arm_sdk_sent']}: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to send {self.stats['arm_sdk_sent']}: {type(e).__name__}: {e}")
            return False


    def send_arm_sdk(self, msg: dict):
        """Send with tracking"""
        if not self.loop or not self.loop.is_running():
            print("ERROR: Event loop not running!")
            return
        
        if not self.websocket:
            print("ERROR: No websocket!")
            return
            
        if self.websocket.closed:
            print(f"ERROR: WebSocket closed before send!")
            return
        
        # Track pending sends
        if self.pending_sends > 10:
            print(f"WARNING: {self.pending_sends} sends still pending!")
        
        try:
            self.pending_sends += 1
            
            future = asyncio.run_coroutine_threadsafe(
                self.send_arm_sdk_async(msg),
                self.loop
            )
            
            # Add callback to track completion
            def done_callback(fut):
                self.pending_sends -= 1
                try:
                    result = fut.result()
                    if not result:
                        self.send_failures += 1
                        print(f"Send failed! Total failures: {self.send_failures}")
                except Exception as e:
                    self.send_failures += 1
                    print(f"Send exception: {e}")
            
            future.add_done_callback(done_callback)
            
        except Exception as e:
            self.pending_sends -= 1
            print(f"ERROR scheduling send: {e}")



    def process_lowstate_data(self, data):
        """Process lowstate data from WebSocket and update buffer"""
        # Create a new lowstate object
        lowstate = G1_23_LowState()
        
        # Parse the motor states from the WebSocket data
        if 'motor_q' in data and 'motor_qd' in data:
            motor_q = data['motor_q']
            motor_qd = data['motor_qd']
            
            # Update motor states
            for i in range(min(len(motor_q), G1_23_Num_Motors)):  # G1_23_Num_Motors = 35
                lowstate.motor_state[i].q = motor_q[i]
                lowstate.motor_state[i].dq = motor_qd[i]
        
        # Update the buffer
        self.lowstate_buffer.SetData(lowstate)
        self.stats['lowstate_received'] += 1

    async def receive_messages(self):
        """Receive messages from the server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'lowstate':
                        self.process_lowstate_data(data['data'])
                    elif msg_type == 'status':
                        print(f"Server status: {data['data']}")
                    elif msg_type == 'stats':
                        print(f"Bridge stats: {data['data']}")
                    else:
                        print(f"Unknown message type: {msg_type}")
                        
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
                except Exception as e:
                    print(f"Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error in receive loop: {e}")

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def start_control(self):
        """Call this when ready to start sending commands"""
        self.ready_to_send = True


    def _ctrl_motor_state(self):
        """Control thread that sends motor commands"""

        loop_count = 0
        send_every_n = 10

        while self.running:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target.copy()
                arm_tauff_target = self.tauff_target.copy()

            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            # Update message dict with arm motor values
            for idx, joint_id in enumerate(G1_23_JointArmIndex):
                self.msg_dict['target_q'][joint_id.value] = float(cliped_arm_q_target[idx])
                self.msg_dict['target_tau'][joint_id.value] = float(arm_tauff_target[idx])

            # Only SEND every Nth iteration
            if loop_count % send_every_n == 0 and self.ready_to_send:
                self.send_arm_sdk(self.msg_dict)

            loop_count += 1

            # Handle gradual speed increase
            if self._speed_gradual_max:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # print(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # print(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_23_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        print("[G1_23_ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(10)
            # self.tauff_target = np.zeros(10)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while True:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                print("[G1_23_ArmController] both arms have reached the home position.")
                break
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def shutdown(self):
        """Shutdown the controller"""
        print("Shutting down controller...")
        self.running = False
        
        # Stop the async loop
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for threads to finish
        if hasattr(self, 'async_thread'):
            self.async_thread.join(timeout=2.0)
        if hasattr(self, 'publish_thread'):
            self.publish_thread.join(timeout=2.0)
        
        print("Controller shutdown complete")

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK
    import pinocchio as pin

    try:

        arm_ik = G1_23_ArmIK(Unit_Test=True, Visualization=False)
        arm = G1_23_ArmController_WS("ws://192.168.1.124:8765")

        # Initial position
        L_tf_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, +0.25, 0.1]),
        )

        R_tf_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, -0.25, 0.1]),
        )

        rotation_speed = 0.005  # Rotation speed in radians per iteration
        step = 0
        
        arm.speed_gradual_max()
        arm.start_control() 
        
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001, 0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001, 0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\nStopping...")
        arm.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        arm.shutdown()