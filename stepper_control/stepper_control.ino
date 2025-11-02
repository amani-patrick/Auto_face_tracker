#include <Arduino.h>
#include <Stepper.h>

// Stepper motor setup
const int stepsPerRevolution = 2048;  // 28BYJ-48 full rotation
Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);
  myStepper.setSpeed(15);  // RPM (adjust if needed)
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    switch (command) {
      case 'L':
        myStepper.step(20);  // Rotate left
        break;
      case 'R':
        myStepper.step(-20); // Rotate right
        break;
      case 'C':
        // Center (optional: do nothing)
        break;
    }
  }
}
