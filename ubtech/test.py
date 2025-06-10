from ubtechapi import YanAPI
import time

robot_ip = '192.168.1.152'
YanAPI.yan_api_init(robot_ip)

YanAPI.open_vision_stream(resolution = '640x480')

# result = YanAPI.sync_do_object_recognition()

# print(result)

# print(YanAPI.get_robot_fall_management_state())
# YanAPI.set_robot_fall_management_state(True)

# YanAPI.sync_play_motion(name='bend',direction='left',speed='normal',repeat=1)

# YanAPI.sync_play_motion(name='bow',speed='normal',repeat=1)

# YanAPI.sync_play_motion(name='reset')

# res = YanAPI.set_servos_angles({'NeckLR':60})
# time.sleep(1)

# res = YanAPI.set_servos_angles({'NeckLR':90})
# time.sleep(1)

# res = YanAPI.set_servos_angles({'NeckLR':120})
# time.sleep(1)

# res = YanAPI.set_servos_angles({'NeckLR':90})
# time.sleep(1)
# print(res)