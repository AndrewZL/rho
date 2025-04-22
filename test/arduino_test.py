from rob498.arduino_interface import ArduinoInterface

if __name__ == "__main__":
    arduino = ArduinoInterface("/dev/ttyUSB0")

    # Try to send the DROP_X command
    if arduino.send_command("DROP_X"):
        print("Command sent. Waiting for confirmation in background.")
    else:
        print("Still waiting for confirmation, command not sent.")

    # Later, check if we can send another command
    arduino.send_command("DROP_X")
