import logging
import threading
import time
import socket
import struct

from enum import Enum

try:
    import hid
except ImportError:
    logging.warning(
        "HID library not found. Please install the HIDAPI library to use this plugin."
    )
    hid = None

from actions.base import ActionConfig, ActionConnector
from actions.move_safe.interface import MoveInput

MAX_RAW_SPEED = 20000
HEARTBEAT_INTERVAL = 0.1

COMMAND_CODES = {
    "stand up":   0x21010202,
    "sit": 0x21010202,
    "shake paw":     0x21010507,   # shake paw
    "dance":   0x21010204,
    "moon walk":   0x2101030C,
    "x_move":   0x21010130,
    "y_move":   0x21010131,
    "rotate": 0x21010135,
    "stand still": 0x21010D05,
    "movement mode": 0x21010D06,
}

class RobotState(Enum):
    STANDING = "standing"
    SITTING = "sitting"

class MoveRos2Connector(ActionConnector[MoveInput]):

    def __init__(self, config: ActionConfig):
        super().__init__(config)

        self.current_state = RobotState.SITTING

        self.joysticks = []

        self.vendor_id = ""
        self.product_id = ""
        self.button_previous = None
        self.d_pad_previous = None
        self.rt_previous = 0
        self.lt_previous = 0
        self.gamepad = None

        self.move_speed = 0.7
        self.turn_speed = 0.6

        if hid is not None:
            for device in hid.enumerate():
                logging.debug(f"device {device['product_string']}")
                if "Xbox Wireless Controller" in device["product_string"]:
                    self.vendor_id = device["vendor_id"]
                    self.product_id = device["product_id"]
                    self.gamepad = hid.Device(self.vendor_id, self.product_id)
                    logging.info(
                        f"Connected {device['product_string']} {self.vendor_id} {self.product_id}"
                    )
                    break

        # create control server
        try:
            self.local_port = 20001
            self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
            self.ctrl_addr = ("192.168.1.120", 43893)
            logging.info("DR server initialized")
        except Exception as e:
            logging.error(f"Error initializing DR server: {e}")

        try:
            self.start_heartbeat()
            logging.info("DR server heartbeat initialized")
        except Exception as e:
            logging.error(f"Error starting heart beat: {e}")

        try:
            self._send_command(COMMAND_CODES["stand up"])
            logging.info("Robot standing")
            self.current_state = RobotState.STANDING
        except Exception as e:
            logging.error(f"Error standing: {e}")

        try:
            self._send_command(COMMAND_CODES["movement mode"])
            logging.info("Robot enters movement mode")
        except Exception as e:
            logging.error(f"Error entering movement mode: {e}")

        self.thread_lock = threading.Lock()

    def _send_command(self, code, param1=0, param2=0):
        logging.info(f"Sent Command={code}, Param1={param1}, Param2={param2}")
        self._send_simple(code, param1, param2)

    def _send_simple(self, code, param1=0, param2=0):
        try:
            payload = struct.pack('<3i', code, param1, param2)
            self.server.sendto(payload, self.ctrl_addr)
        except Exception as e:
            logging.info(f"Error sending command：{e}")

    def start_heartbeat(self):
        self._send_simple(0x21040001)
        self.heartbeat = threading.Timer(HEARTBEAT_INTERVAL, self.start_heartbeat)
        self.heartbeat.start()

    def stop_heartbeat(self):
        if hasattr(self, 'heartbeat'):
            self.heartbeat.cancel()

    def _execute_command_thread(self, command: str) -> None:
        try:
            if command == "stand up" and self.current_state == RobotState.STANDING:
                logging.info("Already standing, skipping command")
                return
            elif command == "sit" and self.current_state == RobotState.SITTING:
                logging.info("Already sitting, skipping command")
                return

            code = COMMAND_CODES[command]
            self._send_command(code)

            if command == "stand still":
                time.sleep(2.0)
                code = COMMAND_CODES["movement mode"]
                self._send_command(code)

            logging.info(f"DR command {command} executed")

            if command == "stand up":
                self.current_state = RobotState.STANDING
            elif command == "sit":
                self.current_state = RobotState.SITTING

        except Exception as e:
            logging.error(f"Error in command thread {command}: {e}")
        finally:
            self.thread_lock.release()

    def _execute_sport_command_sync(self, command: str) -> None:
        if not self.server:
            return

        if not self.thread_lock.acquire(blocking=False):
            logging.info("Action already in progress, skipping")
            return

        try:
            thread = threading.Thread(
                target=self._execute_command_thread, args=(command,), daemon=True
            )
            thread.start()
        except Exception as e:
            logging.error(f"Error executing DR command {command}: {e}")
            self.thread_lock.release()

    async def _execute_sport_command(self, command: str) -> None:
        if not self.server:
            return

        if not self.thread_lock.acquire(blocking=False):
            logging.info("Action already in progress, skipping")
            return

        try:
            thread = threading.Thread(
                target=self._execute_command_thread, args=(command,), daemon=True
            )
            thread.start()
        except Exception as e:
            logging.error(f"Error executing DR command {command}: {e}")
            self.thread_lock.release()

    async def connect(self, output_interface: MoveInput) -> None:

        # This is a limited subset of Go2 movements that are
        # generally safe. Note that the "stretch" action involves
        # about 40 cm of back and forth motion, and the "dance"
        # action involves copious jumping in place for about 10 seconds.
        if output_interface.action == "stand up":
            logging.info("DR command: stand up")
            await self._execute_sport_command("stand up")
        elif output_interface.action == "sit":
            logging.info("DR command: lay down")
            await self._execute_sport_command("sit")
        elif output_interface.action == "shake paw":
            logging.info("DR command: shake paw")
            await self._execute_sport_command("shake paw")
        elif output_interface.action == "moon walk":
            logging.info("DR command: moon walk")
            await self._execute_sport_command("moon walk")
        elif output_interface.action == "dance":
            logging.info("DR command: dance")
            await self._execute_sport_command("dance")
        elif output_interface.action == "stand still":
            logging.info("DR command: stand still")
            await self._execute_sport_command("stand still")
        else:
            logging.info(f"Unknown move type: {output_interface.action}")

        logging.info(f"SendThisToDR: {output_interface.action}")

    def _move_robot(self, move_speed_x, move_speed_y, rotate_speed=0.0) -> None:
        if not self.server or self.current_state != RobotState.STANDING:
            return

        try:
            logging.info(
                f"Moving robot: move_speed_x={move_speed_x}, move_speed_y={move_speed_y}, rotate_speed={rotate_speed}"
            )

            # convert normalized speeds (−1.0…1.0) into raw ints:
            x_raw   = int(move_speed_x  * self.MAX_RAW_SPEED)
            y_raw   = int(move_speed_y  * self.MAX_RAW_SPEED)
            r_raw   = int(rotate_speed * self.MAX_RAW_SPEED)

            # send one UDP packet per axis
            # X‐axis (forward/back)
            self._send_command(
                COMMAND_CODES["x_move"],
                x_raw,    # speed
                0         # unused
            )

            # Y‐axis (strafe left/right)
            self._send_command(
                COMMAND_CODES["y_move"],
                y_raw,
                0
            )

            # rotation (yaw)
            self._send_command(
                COMMAND_CODES["rotate"],
                r_raw,
                0
            )

        except Exception as e:
            logging.error(f"Error moving robot: {e}")

    def tick(self) -> None:

        time.sleep(0.1)