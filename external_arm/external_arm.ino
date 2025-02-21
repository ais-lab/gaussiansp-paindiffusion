#define a 0.8
int analogPin = 1;
int val = 0;
float potention_angle;
float rc = 0;

// Calibration values (measured manually)
const int val_min = 67;   // Value at 0°
const int val_max = 1023; // Value at 150°

void setup() {
  Serial.begin(9600);
}

void loop() {
  val = analogRead(analogPin);

  // Clamp the value to avoid negative angles or overshooting
  val = constrain(val, val_min, val_max);

  // Calculate calibrated angle
  potention_angle = (float)(val - val_min) / (val_max - val_min) * 150.0f;

  // Apply RC filter (optional)
  static bool first = true;
  if (first) {
    rc = potention_angle; // Initialize filter with first reading
    first = false;
  } else {
    rc = a * rc + (1 - a) * potention_angle;
  }

  Serial.println(rc);
  delay(10);
}