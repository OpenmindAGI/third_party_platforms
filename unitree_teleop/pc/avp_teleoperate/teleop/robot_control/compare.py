#!/usr/bin/env python3
"""
WebSocket-CycloneDDS Bridge for Unitree Robot
Adds a high-rate control loop that interpolates from the
robot's current state to the latest target_q/target_tau
received over WebSocket.
"""

import os
import sys
import json
import time
import asyncio
import threading
import queue
from typing import Dict, Any
from enum import IntEnum

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# Unitree SDK imports...
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.utils.crc import CRC

# … (G1_23_JointArmIndex, G1_23_JointIndex, G1JointIndex enums unchanged) …

class WebSocketBridge:
    def __init__(self,
                 ws_host: str = "0.0.0.0",
                 ws_port: int = 8765,
                 eth0_ip: str = "192.168.123.164",
                 control_rate: float = 250.0,
                 interp_duration: float = 0.1):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.eth0_ip = eth0_ip
        self.running = True

        # interpolation parameters
        self.control_rate = control_rate
        self.interp_duration = interp_duration
        self.interp_lock = threading.Lock()
        # initialize with zeros until first target arrives
        self.initial_q   = np.zeros(35)
        self.initial_tau = np.zeros(35)
        self.target_q    = np.zeros(35)
        self.target_tau  = np.zeros(35)
        self.last_cmd_time = time.time()

        # WebSocket state
        self.ws_clients = set()
        self.lowstate_queue = asyncio.Queue()
        self.stats = {'lowstate_sent': 0, 'dds_sent': 0, 'ws_clients': 0}

        # DDS publisher thread / queue
        self.crc = CRC()
        self.command_queue = queue.Queue(maxsize=200)
        self.dds_thread = threading.Thread(target=self._dds_publisher_thread, daemon=True)
        self.dds_thread.start()

        # DDS channels
        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self.on_lowstate_received, 10)

        # wait until we get the first lowstate
        self._have_state = threading.Event()
        self.low_state = None
        self._have_state.wait()

        # start interpolation/control thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def on_lowstate_received(self, msg: LowState_):
        """Receive hardware state and queue it out to any WS clients."""
        self.low_state = msg
        if not self._have_state.is_set():
            self._have_state.set()

        data = {
            'motor_q': [m.q for m in msg.motor_state],
            'motor_qd': [m.dq for m in msg.motor_state]
        }
        # broadcast every time
        asyncio.run_coroutine_threadsafe(
            self.lowstate_queue.put({'type':'lowstate','data':data,'ts':time.time()}),
            asyncio.get_event_loop()
        )
        self.stats['lowstate_sent'] += 1

    def _dds_publisher_thread(self):
        """Publish LowCmd_ messages from self.command_queue onto DDS."""
        while self.running:
            try:
                msg_dict = self.command_queue.get(timeout=0.005)
            except queue.Empty:
                continue

            cmd = self.dict_to_lowcmd(msg_dict)
            cmd.crc = self.crc.Crc(cmd)
            self.lowcmd_pub.Write(cmd)
            self.stats['dds_sent'] += 1

    def dict_to_lowcmd(self, data: Dict[str, Any]) -> LowCmd_:
        msg = LowCmd_()
        # set per‐motor fields
        for i in range(len(data['target_q'])):
            msg.motor_cmd[i].q   = data['target_q'][i]
            msg.motor_cmd[i].tau = data['target_tau'][i]
            # leave kp/kd as you had previously…
        return msg

    async def handle_websocket(self, websocket: WebSocketServerProtocol):
        self.ws_clients.add(websocket)
        self.stats['ws_clients'] = len(self.ws_clients)
        # initial status
        await websocket.send(json.dumps({'type':'status','data':{'clients':self.stats}}))

        try:
            async for raw in websocket:
                msg = json.loads(raw)
                if msg.get('type') == 'lowcmd':
                    payload = msg['data']
                    # grab current hardware state as the start of interp
                    with self.interp_lock:
                        self.initial_q   = np.array([m.q for m in self.low_state.motor_state])
                        self.initial_tau = np.array(self.target_tau)  # could also grab current DDS‐sent tau
                        self.target_q    = np.array(payload['target_q'])
                        self.target_tau  = np.array(payload['target_tau'])
                        self.last_cmd_time = time.time()
                # ignore other WS types…
        finally:
            self.ws_clients.remove(websocket)
            self.stats['ws_clients'] = len(self.ws_clients)

    async def broadcast_lowstate(self):
        """Send lowstate JSON blobs out to every open WS connection."""
        while self.running:
            try:
                msg = await asyncio.wait_for(self.lowstate_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            dead = set()
            for ws in self.ws_clients:
                try:
                    await ws.send(json.dumps(msg))
                except:
                    dead.add(ws)
            self.ws_clients -= dead
            self.stats['ws_clients'] = len(self.ws_clients)

    async def print_stats(self):
        while self.running:
            await asyncio.sleep(5)
            print(f"[Bridge] WS-clients={self.stats['ws_clients']}  DDS-sent={self.stats['dds_sent']}")

    async def run_websocket_server(self):
        # start WS + background tasks
        broadcast = asyncio.create_task(self.broadcast_lowstate())
        stats_printer = asyncio.create_task(self.print_stats())
        async with websockets.serve(self.handle_websocket, self.ws_host, self.ws_port):
            await asyncio.Future()  # run forever

        broadcast.cancel()
        stats_printer.cancel()

    def _control_loop(self):
        """High-rate loop: linearly interpolate q/tau and push to DDS."""
        dt = 1.0 / self.control_rate
        while self.running:
            with self.interp_lock:
                t = time.time() - self.last_cmd_time
                alpha = min(1.0, t / self.interp_duration)
                q_cmd   = self.initial_q   + (self.target_q   - self.initial_q)   * alpha
                tau_cmd = self.initial_tau + (self.target_tau - self.initial_tau) * alpha

            # push into the DDS‐publisher queue
            try:
                self.command_queue.put_nowait({
                    'target_q':   q_cmd.tolist(),
                    'target_tau': tau_cmd.tolist()
                })
            except queue.Full:
                # if you get here, DDS can't keep up even at your control_rate
                pass

            time.sleep(dt)

    def run(self):
        print(f"Starting bridge on WS {self.ws_host}:{self.ws_port}, "
              f"ctrl @ {self.control_rate} Hz, interp {self.interp_duration}s")
        ChannelFactoryInitialize(0, 'eth0')
        asyncio.run(self.run_websocket_server())

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ws-host', default='0.0.0.0')
    p.add_argument('--ws-port', type=int, default=8765)
    p.add_argument('--eth0-ip', default='192.168.123.164')
    p.add_argument('--control-rate', type=float, default=250.0,
                   help='Control loop frequency in Hz')
    p.add_argument('--interp-duration', type=float, default=0.1,
                   help='Time (s) over which to interpolate to each new target')
    args = p.parse_args()

    bridge = WebSocketBridge(
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        eth0_ip=args.eth0_ip,
        control_rate=args.control_rate,
        interp_duration=args.interp_duration
    )
    bridge.run()

if __name__ == "__main__":
    main()
