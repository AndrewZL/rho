#include <Servo.h>

const int redPin = 10;
const int greenPin = 11;
const int bluePin = 6;
const int servoPin = 9;

const int DEFAULT_ANGLE = 40;
const int ALIGNED_ANGLE = 168;
const int ALIGNED_DELAY = 200; // Delay at aligned angle (milliseconds)

Servo myServo;

// Define two states: idle (flashing) and executing a command.
enum State {
  IDLE,
  EXECUTING_DROP
};

State currentState = IDLE;

// Helper function to set the LED color.
// For a common anode LED, we invert the brightness.
void setColor(uint8_t r, uint8_t g, uint8_t b) {
  analogWrite(redPin, 255 - r);
  analogWrite(greenPin, 255 - g);
  analogWrite(bluePin, 255 - b);
}

// Function to execute the DROP_X command.
void executeDropX() {
  // Move servo to the aligned position.
  myServo.write(ALIGNED_ANGLE);
  delay(200);  // Allow time for servo to reach aligned position
  
  // Set LED to green while at the aligned angle.
  setColor(255, 0, 0);
  delay(ALIGNED_DELAY);
  
  // Set LED back to blue while moving the servo back.
  setColor(0, 0, 255);
  myServo.write(DEFAULT_ANGLE);
  delay(200);  // Allow time for servo to return
  
  // Send confirmation back to the Jetson Nano.
  Serial.println("DROP_X_CONFIRM");
  
  // Return to idle state.
  currentState = IDLE;
}

void setup() {
  Serial.begin(115200);
  myServo.attach(servoPin);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  
  // Initialize servo position.
  myServo.write(DEFAULT_ANGLE);
}

void loop() {
  if (currentState == IDLE) {
    // Check for incoming serial commands.
    if (Serial.available() > 0) {
      String command = Serial.readStringUntil('\n');
      command.trim();
      if (command.startsWith("DROP_X")) {
        currentState = EXECUTING_DROP;
        // Immediately indicate command reception with a solid blue LED.
        setColor(0, 0, 255);
        executeDropX();
      }
      // Future commands can be added here.
    } else {
      // Idle behavior: flash white with a 10% duty cycle at 1Hz.
      unsigned long currentMillis = millis();
      if ((currentMillis % 1000) < 100) { // First 100ms of every second
        setColor(255, 255, 255); // White (all channels full brightness)
      } else {
        setColor(0, 0, 0); // LED off
      }
    }
  }
  // When in EXECUTING_DROP state, the execution function handles LED updates.
}
