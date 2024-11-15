import os
import time
import busio
import digitalio
import board
print(dir(board.pin))
button = digitalio.DigitalInOut(board.D12) # D24 for kill switch, D12 for skip
button.switch_to_input(pull=digitalio.Pull.DOWN)
while(True):
    print(button.value)
    time.sleep(0.5)