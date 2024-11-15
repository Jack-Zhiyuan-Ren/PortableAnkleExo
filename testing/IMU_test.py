import time
import board
import adafruit_lsm9ds1

# Create sensor object, communicating over the board's default I2C bus
#i2c = board.I2C()  # uses board.SCL and board.SDA
#sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)

# SPI connection:
# from digitalio import DigitalInOut, Direction
# spi = board.SPI()
# csag = DigitalInOut(board.D5)
# csag.direction = Direction.OUTPUT
# csag.value = True
# csm = DigitalInOut(board.D6)
# csm.direction = Direction.OUTPUT
# csm.value = True

spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
cs = digitalio.DigitalInOut(board.D17)
csm = digitalio.DigitalInOut(baord.D16) # may not need?
sensor = adafruit_lsm9ds1.LSM9DS1_SPI(spi, cs, csm)
sensor.accel_range(ACCELRANGE_8G) # also has 16G
sensor.gyro_scale(GYROSCALE_500DPS) # also has 2000DPS
# Main loop will read the acceleration, magnetometer, gyroscope, Temperature
# values every second and print them out.
while True:
    # Read acceleration, magnetometer, gyroscope, temperature.
    accel_x, accel_y, accel_z = sensor.acceleration
    mag_x, mag_y, mag_z = sensor.magnetic
    gyro_x, gyro_y, gyro_z = sensor.gyro
    temp = sensor.temperature
    # Print values.
    print(
        "Acceleration (m/s^2): ({0:0.3f},{1:0.3f},{2:0.3f})".format(
            accel_x, accel_y, accel_z
        )
    )
    print(
        "Magnetometer (gauss): ({0:0.3f},{1:0.3f},{2:0.3f})".format(mag_x, mag_y, mag_z)
    )
    print(
        "Gyroscope (rad/sec): ({0:0.3f},{1:0.3f},{2:0.3f})".format(
            gyro_x, gyro_y, gyro_z
        )
    )
    print("Temperature: {0:0.3f}C".format(temp))
    # Delay for a second.
    time.sleep(1.0)