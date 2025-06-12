#!/usr/bin/env python3
"""
WebSocket-CycloneDDS Bridge for Unitree Robot
Communicates with controller via CycloneDDS on eth0 and with PC via WebSocket
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from enum import IntEnum
import queue
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("Error: websockets not installed. Install with: pip install websockets")
    sys.exit(1)

# Import Unitree SDK
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC


kPi = 3.141592654
kPi_2 = 1.57079632


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

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

class WebSocketBridge:
    def __init__(self, ws_host: str = "0.0.0.0", ws_port: int = 8765, eth0_ip: str = "192.168.123.164"):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.eth0_ip = eth0_ip
        self.running = True
        
        # WebSocket clients
        self.ws_clients = set()
        
        # Message queues
        self.lowstate_queue = asyncio.Queue()
        self.lowcmd_queue = asyncio.Queue()
        self.arm_sdk_queue = asyncio.Queue()
        self.crc = CRC()
        # self.low_cmd = unitree_hg_msg_dds__LowCmd_()  

        # Queue for DDS publishing (separate thread)
        self.command_queue = queue.Queue(maxsize=100)
        self.dds_thread = threading.Thread(target=self._dds_publisher_thread)
        self.dds_thread.daemon = True
        self.dds_thread.start()
        
        # Statistics
        self.stats = {
            'lowstate_sent': 0,
            'lowcmd_received': 0,
            'arm_sdk_received': 0,
            'ws_clients': 0
        }

        self.first_update_low_state = False

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.on_lowstate_received, 10)

        self.arm_joints = [
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,
          G1JointIndex.WaistYaw,
          G1JointIndex.WaistRoll,
          G1JointIndex.WaistPitch
        ]
        # self.kp = 60.
        # self.kd = 1.5

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5
        self.all_motor_q = None

        self.dq = 0.
        self.tau = 0.
        self.time_ = 0.0
        self.duration_ = 5.0
        self.low_state = None
        self.lowstate_counter = 0

        while not self.first_update_low_state:
            time.sleep(0.1)
        self.mode_machine_ = self.low_state.mode_machine
        print("Getting Low State")

    def _dds_publisher_thread(self):
        """Dedicated thread for all DDS publishing"""
        print("DDS publisher thread started")
        while self.running:
            try:
                # Get message with timeout
                item = self.command_queue.get(timeout=0.001)
                msg_type, data = item
                
                if self.command_queue.qsize() > 0:
                    print(f"Queue size: {self.command_queue.qsize()}")
            
                try:
                    if msg_type == 'arm_sdk':
                        # Process and publish
                        arm_cmd = self.dict_to_lowcmd(data)
                        self.arm_sdk_publisher.Write(arm_cmd)
                        self.stats['arm_sdk_received'] += 1
                        
                    elif msg_type == 'lowcmd':
                        low_cmd = self.dict_to_lowcmd(data)
                        self.lowcmd_publisher.Write(low_cmd)
                        self.stats['lowcmd_received'] += 1
                        
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {msg_type}: {e}")
                    if error_count < 5:  # Only print traceback for first few errors
                        import traceback
                        traceback.print_exc()
                        
            except queue.Empty:
                # This is normal - no messages to process
                continue
            except Exception as e:
                print(f"DDS publisher thread error: {e}")
                import traceback
                traceback.print_exc()

    def on_lowstate_received(self, msg: LowState_):
        """Callback when LowState is received from controller"""
        self.lowstate_counter = self.lowstate_counter + 1
        if self.lowstate_counter % 10 != 0:  # Only send every 50th
            return

        # Convert message to dict for JSON serialization
        self.low_state = msg
        msg_dict = self.lowstate_to_dict(msg)

        if self.first_update_low_state == False:
            self.first_update_low_state = True

        # Put in queue for WebSocket transmission
        try:
            self.lowstate_queue.put_nowait({
                'type': 'lowstate',
                'data': msg_dict,
                'timestamp': time.time()
            })
            self.stats['lowstate_sent'] += 1
        except asyncio.QueueFull:
            print("Warning: LowState queue full, dropping message")
            
    def lowstate_to_dict(self, msg: LowState_) -> Dict[str, Any]:
        """Convert LowState message to dictionary"""
        return {
            'motor_q': [msg.motor_state[i].q for i in range(35)],
            'motor_qd': [msg.motor_state[i].dq for i in range(35)]
        }

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_23_JointIndex.kLeftAnklePitch.value,
            G1_23_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_23_JointIndex.kLeftShoulderPitch.value,
            G1_23_JointIndex.kLeftShoulderRoll.value,
            G1_23_JointIndex.kLeftShoulderYaw.value,
            G1_23_JointIndex.kLeftElbow.value,
            # Right arm
            G1_23_JointIndex.kRightShoulderPitch.value,
            G1_23_JointIndex.kRightShoulderRoll.value,
            G1_23_JointIndex.kRightShoulderYaw.value,
            G1_23_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
        
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors

    def dict_to_lowcmd(self, data: Dict[str, Any]) -> LowCmd_:
        """Convert dictionary to LowCmd message"""
        msg = unitree_hg_msg_dds__LowCmd_()

        # Fill in the message fields from the dictionary
        if 'target_q' in data:
            # print("Moving")
            msg.mode_pr = 0
            arm_indices = set(member.value for member in G1_23_JointArmIndex)
            for id in G1_23_JointIndex:
                if id.value in arm_indices:
                    if self._Is_wrist_motor(id):
                        msg.motor_cmd[id].kp = self.kp_wrist
                        msg.motor_cmd[id].kd = self.kd_wrist
                    else:
                        msg.motor_cmd[id].kp = self.kp_low
                        msg.motor_cmd[id].kd = self.kd_low
                else:
                    if self._Is_weak_motor(id):
                        msg.motor_cmd[id].kp = self.kp_low
                        msg.motor_cmd[id].kd = self.kd_low
                    else:
                        msg.motor_cmd[id].kp = self.kp_high
                        msg.motor_cmd[id].kd = self.kd_high
                # msg.motor_cmd[id].q  = self.all_motor_q
                
            msg.mode_machine = self.mode_machine_
            msg.motor_cmd[G1JointIndex.kNotUsedJoint].q =  1
            for i, joint in enumerate(self.arm_joints):
                msg.motor_cmd[joint].mode =  1
                msg.motor_cmd[joint].tau = data['target_tau'][joint]
                msg.motor_cmd[joint].q = data['target_q'][joint]
                # msg.motor_cmd[joint].q = (1.0 - ratio) * self.low_state.motor_state[joint].q 
                msg.motor_cmd[joint].dq = self.dq
                # msg.motor_cmd[joint].kp = self.kp 
                # msg.motor_cmd[joint].kd = self.kd

        msg.crc = self.crc.Crc(msg)
        # Add more fields as needed
        return msg

    async def handle_websocket(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket with detailed debugging"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"\n[{time.strftime('%H:%M:%S')}] New WebSocket client connected: {client_id}")
        self.ws_clients.add(websocket)
        self.stats['ws_clients'] = len(self.ws_clients)
        
        msg_count = 0
        last_msg_time = time.time()
        
        try:
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'status',
                'data': {
                    'connected': True,
                    'eth0_ip': self.eth0_ip,
                    'stats': self.stats
                }
            }))
            print(f"[{client_id}] Sent initial status")
            
            # Handle incoming messages
            async for message in websocket:
                msg_count += 1
                current_time = time.time()
                time_since_last = current_time - last_msg_time
                last_msg_time = current_time
                
                # Log message frequency
                if msg_count <= 10 or msg_count % 50 == 0:
                    print(f"[{client_id}] Message #{msg_count}, gap: {time_since_last:.3f}s")
                
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type in ['arm_sdk', 'lowcmd']:
                        # Check queue status before adding
                        queue_size = self.command_queue.qsize()
                        if queue_size > 50:
                            print(f"[{client_id}] WARNING: Queue getting full: {queue_size}/100")
                        
                        try:
                            self.command_queue.put_nowait((msg_type, data.get('data', {})))
                        except queue.Full:
                            print(f"[{client_id}] ERROR: Queue full, dropping {msg_type} message")
                            # Send feedback to client
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'data': 'Queue full, message dropped'
                            }))
                    
                    elif msg_type == 'ping':
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': time.time()
                        }))
                        
                except json.JSONDecodeError:
                    print(f"[{client_id}] Invalid JSON received")
                except Exception as e:
                    print(f"[{client_id}] Error handling message: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # If we get here, the connection closed normally
            print(f"\n[{client_id}] Connection closed normally after {msg_count} messages")
            
        except websockets.exceptions.ConnectionClosed as e:
            print(f"\n[{client_id}] Connection closed: code={e.code}, reason={e.reason}")
            print(f"[{client_id}] Received {msg_count} messages before close")
        except Exception as e:
            print(f"\n[{client_id}] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.ws_clients.discard(websocket)
            self.stats['ws_clients'] = len(self.ws_clients)
            print(f"[{client_id}] Client removed. Active clients: {self.stats['ws_clients']}")


    async def broadcast_lowstate(self):
        """Broadcast LowState messages to all WebSocket clients"""
        while self.running:
            try:
                # Get message from queue with timeout
                msg = await asyncio.wait_for(self.lowstate_queue.get(), timeout=0.1)
                
                if self.ws_clients:
                    # Broadcast to all connected clients
                    message = json.dumps(msg)
                    disconnected_clients = set()
                    
                    for client in self.ws_clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                            
                    # Remove disconnected clients
                    self.ws_clients -= disconnected_clients
                    self.stats['ws_clients'] = len(self.ws_clients)
                    
            except asyncio.TimeoutError:
                # No messages in queue
                pass
            except Exception as e:
                print(f"Error broadcasting lowstate: {e}")
                
    async def print_stats(self):
        """Print statistics periodically"""
        while self.running:
            await asyncio.sleep(5)
            print(f"Bridge Stats - Clients: {self.stats['ws_clients']}, "
                  f"LowState sent: {self.stats['lowstate_sent']}, "
                  f"LowCmd received: {self.stats['lowcmd_received']}, "
                  f"ArmSDK received: {self.stats['arm_sdk_received']}")
            
    async def run_websocket_server(self):
        """Run the WebSocket server"""
        print(f"Starting WebSocket server on {self.ws_host}:{self.ws_port}")
        
        # Start the broadcast task
        broadcast_task = asyncio.create_task(self.broadcast_lowstate())
        stats_task = asyncio.create_task(self.print_stats())
        
        # Start WebSocket server
        async with websockets.serve(self.handle_websocket, self.ws_host, self.ws_port):
            print(f"WebSocket server listening on ws://{self.ws_host}:{self.ws_port}")
            print("Waiting for connections...")
            
            try:
                await asyncio.Future()  # Run forever
            except asyncio.CancelledError:
                pass
            finally:
                broadcast_task.cancel()
                stats_task.cancel()
                await broadcast_task
                await stats_task
                
    def run(self):
        """Main run method"""
        print("Starting WebSocket-CycloneDDS Bridge...")
        print(f"CycloneDDS on eth0: {self.eth0_ip}")
        print(f"WebSocket on: ws://{self.ws_host}:{self.ws_port}")
        
        try:
            # Run the async WebSocket server
            asyncio.run(self.run_websocket_server())
        except KeyboardInterrupt:
            print("\nShutting down bridge...")
            self.running = False
            
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='WebSocket-CycloneDDS Bridge for Unitree Robot')
    parser.add_argument('--ws-host', default='0.0.0.0', 
                       help='WebSocket server host (default: 0.0.0.0)')
    parser.add_argument('--ws-port', type=int, default=8765,
                       help='WebSocket server port (default: 8765)')
    parser.add_argument('--eth0-ip', default='192.168.123.164',
                       help='IP address of eth0 interface')
    
    args = parser.parse_args()

    ChannelFactoryInitialize(0, 'eth0')

    # Create and run bridge
    bridge = WebSocketBridge(
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        eth0_ip=args.eth0_ip
    )
    bridge.run()
    
if __name__ == "__main__":
    main()