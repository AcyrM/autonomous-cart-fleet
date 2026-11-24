// Cart firmware for WEMOS LOLIN32 Lite (original ESP32, Bluetooth Classic)
// Matches production/bt_controller.py on the PC side.
//
// Protocol (ASCII lines over Bluetooth Classic SPP, '\n' terminated):
//   PC -> cart:
//     DRIVE:<left>:<right>   each -255..255 (negative = backward)
//     RAW:<left>:<right>     same, but without the breakaway boost: exactly this
//                            PWM (soft start still applies). The PC's heading
//                            calibration uses it to measure the lowest PWM that
//                            starts the cart; the PC then rescales its DRIVE
//                            speeds per cart, so nothing here is tuned to one
//                            set of motors / driver / battery / load
//     STOP
//     PING                   keep-alive; answered with PONG, motors untouched
//   cart -> PC:
//     HB:<uptime s>:<reset>:<stalls>:<longest ms>
//                            heartbeat every HEARTBEAT_MS while connected (and
//                            right after connecting); <reset> is why the ESP32
//                            last restarted — BROWNOUT means the motors pulled
//                            the supply voltage down far enough to reset it.
//                            <stalls>: times since boot the main loop froze for
//                            more than STALL_MS (Bluetooth stack blocked, e.g.
//                            radio interference); <longest ms>: the worst one
//     PONG                   reply to PING
// Manual test commands (e.g. from a phone serial app, send with newline):
//   F, B, L, R, S
//
// Safety vs. connection:
//   - Motors stop if no DRIVE/STOP arrives for CMD_TIMEOUT_MS. This only stops
//     the motors — the Bluetooth link stays open, so the cart can sit idle
//     for any length of time and resume as soon as the PC sends DRIVE again.
//     The check runs in its own task (safetyTask), NOT in loop(): when the
//     radio link stalls, a Bluetooth call can block for seconds, and a
//     watchdog inside loop() would stall with it while the cart keeps driving.
//   - loop() never writes to Bluetooth itself: outgoing lines go through a
//     queue to txTask, so a stalled link can't freeze command handling either.
//   - Motors stop if the client disconnects or an unknown command arrives.
//   - The heartbeat keeps traffic flowing on the link during idle periods, so
//     the PC adapter/driver does not drop it as inactive, and it lets the PC
//     notice a dead link quickly.
//
// LOLIN32 Lite notes:
//   - GPIO 22 is also the on-board LED (active low): it will glow while the
//     motor on that pin is driven. Harmless.
//   - GPIO 5 is a boot strapping pin and briefly toggles at power-up, so that
//     motor may twitch at boot.
//   - The board has no 5 V output. Power the motor driver from the battery
//     pack, not from the board, and share GND. Motor start-up current dips can
//     brown-out the ESP32, which resets it and drops the Bluetooth link.

#include "BluetoothSerial.h"
#include "esp_system.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error "Bluetooth Classic not enabled - select board 'WEMOS LOLIN32 Lite' (ESP32)"
#endif

// ── Configuration ─────────────────────────────────────────────────────────────
#define BT_NAME "Cart-1"                 // give each cart a unique name

const uint32_t CMD_TIMEOUT_MS = 500;     // stop motors if the PC goes quiet this long
const uint32_t HEARTBEAT_MS   = 2000;    // keep-alive message interval
const uint32_t SAFETY_TICK_MS = 20;      // safety task check interval
const uint32_t STALL_MS       = 200;     // a loop() pass longer than this = a stall
const int      TX_QUEUE_LEN   = 8;       // outgoing lines waiting for txTask
// No minimum-PWM setting here: the PC measures each cart's start PWM during
// heading calibration and sends speeds that already account for it.
const uint32_t RAMP_MS        = 60;      // soft start: 0 → 255 takes this long. Softens
                                         // the current spike of all four motors
                                         // starting at once (the ESP32 now has its own
                                         // power bank, so this can be short).
                                         // Slowing down / stopping is immediate.
const uint32_t BOOST_MS       = 450;     // breakaway: a side starting from rest (or
                                         // reversing) is driven toward full power for
                                         // this long (ramped, see RAMP_MS), then drops
                                         // to the requested speed — static friction
                                         // needs more torque than keeping it rolling
const int      TEST_SPEED     = 230;     // speed for the F/B/L/R test commands
// Motor PWM frequency. The core's default is 1 kHz (the audible whine): each
// pulse then lets the motor current rise and collapse, and a stalled motor
// turns that into large current spikes on the wires — electrical noise that
// can jam the cart's Bluetooth. At 20 kHz the motor's own inductance smooths
// the current into a near-steady flow (and it's silent). MX1508 / DRV8833 /
// TB6612 drivers handle 20 kHz fine; an L298N switches slowly and gets weaker
// and hotter at high frequency — if that's the driver, try 5000 or go back
// to 1000. After changing this, re-run 🧭 calibration: the start PWM changes.
const uint32_t PWM_FREQ_HZ    = 20000;

struct Motor { uint8_t in1, in2; };

// Left/right grouping inferred from the old 'L'/'R' commands
// (L: M1 fwd, M2 back, M3 back, M4 fwd). VERIFY with the test commands:
// 'F' must drive straight forward, 'L' must spin left in place.
// If one wheel spins the wrong way, swap its two pins here.
Motor LEFT_MOTORS[]  = { {23, 18},    // old Motor2
                         {17,  5} };  // old Motor3
Motor RIGHT_MOTORS[] = { {16,  4},    // old Motor1
                         {22, 19} };  // old Motor4

// ── State ─────────────────────────────────────────────────────────────────────
BluetoothSerial SerialBT;

char     lineBuf[32];
size_t   lineLen     = 0;
bool     overflow    = false;
volatile uint32_t lastCmdMs = 0;          // shared with safetyTask
uint32_t lastHbMs    = 0;
volatile bool driving = false;            // shared with safetyTask
volatile bool safetyStopped = false;      // safetyTask stopped the motors (for the log)
uint32_t lastLoopMs  = 0;
uint32_t stallCount  = 0, stallLongestMs = 0;

// Motor state is changed from loop() and from safetyTask: this mutex keeps
// them from interleaving (never held while touching Bluetooth)
SemaphoreHandle_t motorMutex;
QueueHandle_t     txQueue;
struct TxLine { char s[48]; };
bool     hadClient   = false;
int      curLeft     = 0, curRight = 0;  // requested speeds
float    outLeft     = 0, outRight = 0;  // speeds actually applied (ramping)
int      appliedLeft = 0, appliedRight = 0;
uint32_t lastRampMs  = 0;
uint32_t boostLeftUntil = 0, boostRightUntil = 0;   // 0 = no breakaway boost

// ── Motors ────────────────────────────────────────────────────────────────────
void setMotor(const Motor &m, int speed) {
  int pwm = constrain(abs(speed), 0, 255);
  // analogWrite on both pins: digitalWrite on a pin already attached to
  // PWM (LEDC) is unreliable on the ESP32 core.
  if (speed > 0)      { analogWrite(m.in1, pwm); analogWrite(m.in2, 0);   }
  else if (speed < 0) { analogWrite(m.in1, 0);   analogWrite(m.in2, pwm); }
  else                { analogWrite(m.in1, 0);   analogWrite(m.in2, 0);   }
}

// Move one side's applied speed toward its target: speeding up is limited to
// `step`; slowing down and stopping are immediate; a reversal cuts to 0 first
// and then ramps up the other way (reversing at full power is the worst case
// for current draw)
void stepToward(float &out, int target, float step) {
  if (target == 0 || (out != 0 && (target > 0) != (out > 0))) out = 0;
  if (target == 0) return;
  if (abs(target) <= fabsf(out)) { out = target; return; }
  out += target > 0 ? step : -step;
  if (fabsf(out) > abs(target)) out = target;
}

// Target for one side: full power in the requested direction while its
// breakaway boost runs, the requested speed afterwards
int boosted(int target, uint32_t &until, uint32_t t) {
  if (until != 0 && (int32_t)(t - until) >= 0) until = 0;      // boost over
  if (target != 0 && until != 0) return target > 0 ? 255 : -255;
  return target;
}

// Called on every command and every loop pass: advances the soft start and
// writes the pins only when an output actually changes. Caller holds motorMutex.
void applyMotorsLocked() {
  uint32_t t = millis();
  float step = 255.0f * (t - lastRampMs) / RAMP_MS;
  lastRampMs = t;
  stepToward(outLeft,  boosted(curLeft,  boostLeftUntil,  t), step);
  stepToward(outRight, boosted(curRight, boostRightUntil, t), step);
  int l = (int)outLeft, r = (int)outRight;
  if (l != appliedLeft)  { for (const Motor &m : LEFT_MOTORS)  setMotor(m, l); appliedLeft  = l; }
  if (r != appliedRight) { for (const Motor &m : RIGHT_MOTORS) setMotor(m, r); appliedRight = r; }
  driving = (curLeft != 0 || curRight != 0);
}

void applyMotors() {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  applyMotorsLocked();
  xSemaphoreGive(motorMutex);
}

// A side starting from rest or reversing gets a breakaway boost. The PC
// repeats DRIVE every camera frame: same direction → no new boost.
bool starts(int now, int before) {
  return now != 0 && (before == 0 || (now > 0) != (before > 0));
}

// boost = false (RAW command): exactly the requested PWM, no breakaway boost
void setDriveLocked(int left, int right, bool boost) {
  uint32_t t = millis();
  if (boost && starts(left, curLeft))   boostLeftUntil  = (t + BOOST_MS) | 1;   // never 0
  if (boost && starts(right, curRight)) boostRightUntil = (t + BOOST_MS) | 1;
  if (!boost || left == 0)  boostLeftUntil  = 0;
  if (!boost || right == 0) boostRightUntil = 0;
  curLeft  = left;
  curRight = right;
  applyMotorsLocked();
}

void setDrive(int left, int right, bool boost) {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  setDriveLocked(left, right, boost);
  xSemaphoreGive(motorMutex);
}

void drive(int left, int right) { setDrive(left, right, true); }

void stopAll() { drive(0, 0); }

// ── Safety task ───────────────────────────────────────────────────────────────
// Command watchdog, independent of loop() and of Bluetooth: stops the motors
// when no DRIVE/STOP has arrived for CMD_TIMEOUT_MS, even while loop() is
// stuck inside the Bluetooth stack. Only stops the motors — the link stays.
void safetyTask(void *) {
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(SAFETY_TICK_MS));
    if (driving && millis() - lastCmdMs > CMD_TIMEOUT_MS) {
      xSemaphoreTake(motorMutex, portMAX_DELAY);
      if (driving && millis() - lastCmdMs > CMD_TIMEOUT_MS) {   // re-check under lock
        setDriveLocked(0, 0, true);
        safetyStopped = true;
      }
      xSemaphoreGive(motorMutex);
    }
  }
}

// ── Bluetooth output ──────────────────────────────────────────────────────────
// loop() only queues lines (never waits); txTask does the actual writing and
// is the only thing that blocks when the radio link stalls. A full queue
// drops the line: heartbeats and PONGs are disposable, the PC re-asks.
void sendLine(const char *s) {
  TxLine m;
  snprintf(m.s, sizeof(m.s), "%s\n", s);
  xQueueSend(txQueue, &m, 0);
}

void txTask(void *) {
  TxLine m;
  for (;;) {
    if (xQueueReceive(txQueue, &m, portMAX_DELAY) != pdTRUE) continue;
    if (!SerialBT.hasClient()) continue;   // never write to nobody
    SerialBT.write((const uint8_t *)m.s, strlen(m.s));
  }
}

const char *resetReason() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:   return "POWERON";
    case ESP_RST_BROWNOUT:  return "BROWNOUT";
    case ESP_RST_PANIC:     return "PANIC";
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT:       return "WATCHDOG";
    case ESP_RST_SW:        return "SOFTWARE";
    case ESP_RST_EXT:       return "EXTERNAL";
    default:                return "OTHER";
  }
}

void sendHeartbeat() {
  char msg[48];
  snprintf(msg, sizeof(msg), "HB:%lu:%s:%lu:%lu", (unsigned long)(millis() / 1000),
           resetReason(), (unsigned long)stallCount, (unsigned long)stallLongestMs);
  sendLine(msg);
}

// ── Commands ──────────────────────────────────────────────────────────────────
void handleLine(const char *s) {
  int l, r;
  if (strcmp(s, "PING") == 0) {
    sendLine("PONG");                  // keep-alive only: motor state unchanged
    return;
  }
  // Before starting the motors: otherwise safetyTask could see the new
  // "driving" with the old timestamp and stop the cart at once
  lastCmdMs = millis();
  if (strcmp(s, "STOP") == 0 || strcmp(s, "S") == 0) {
    stopAll();
  } else if (sscanf(s, "DRIVE:%d:%d", &l, &r) == 2) {
    drive(constrain(l, -255, 255), constrain(r, -255, 255));
  } else if (sscanf(s, "RAW:%d:%d", &l, &r) == 2) {
    setDrive(constrain(l, -255, 255), constrain(r, -255, 255), false);
  } else if (strcmp(s, "F") == 0) {
    drive(TEST_SPEED, TEST_SPEED);
  } else if (strcmp(s, "B") == 0) {
    drive(-TEST_SPEED, -TEST_SPEED);
  } else if (strcmp(s, "L") == 0) {
    drive(-TEST_SPEED, TEST_SPEED);
  } else if (strcmp(s, "R") == 0) {
    drive(TEST_SPEED, -TEST_SPEED);
  } else {
    stopAll();                         // unknown -> fail safe (link stays up)
    Serial.printf("Unknown command: '%s'\n", s);
  }
}

void readCommands() {
  while (SerialBT.available()) {
    char c = (char)SerialBT.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (!overflow && lineLen > 0) handleLine(lineBuf);
      lineLen  = 0;
      overflow = false;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      overflow = true;                 // drop over-long garbage line
    }
  }
}

// ── Arduino entry points ──────────────────────────────────────────────────────
// Attach the pin to PWM at 0 (analogWriteFrequency only works on an attached
// pin in core 3.x), then set PWM_FREQ_HZ
void setupPwmPin(uint8_t pin) {
  pinMode(pin, OUTPUT);
  analogWrite(pin, 0);
  analogWriteFrequency(pin, PWM_FREQ_HZ);
}

void setup() {
  Serial.begin(115200);
  for (const Motor &m : LEFT_MOTORS)  { setupPwmPin(m.in1); setupPwmPin(m.in2); }
  for (const Motor &m : RIGHT_MOTORS) { setupPwmPin(m.in1); setupPwmPin(m.in2); }
  motorMutex = xSemaphoreCreateMutex();
  txQueue    = xQueueCreate(TX_QUEUE_LEN, sizeof(TxLine));
  stopAll();
  SerialBT.begin(BT_NAME);             // stays discoverable/connectable after a drop
  // Safety above loop() (priority 1) so it runs even if loop() spins; both on
  // the app core, the Bluetooth stack runs on the other one
  xTaskCreatePinnedToCore(safetyTask, "safety", 2048, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(txTask,     "bt_tx",  4096, nullptr, 1, nullptr, 1);
  Serial.printf("%s ready, MAC %s, last reset: %s\n", BT_NAME,
                SerialBT.getBtAddressString().c_str(), resetReason());
  lastLoopMs = millis();
}

void loop() {
  uint32_t now = millis();
  // Stall detection: a loop pass that took far longer than its 1 ms delay
  // means a Bluetooth call blocked (link stalled / radio jammed)
  uint32_t gap = now - lastLoopMs;
  lastLoopMs = now;
  if (gap > STALL_MS) {
    stallCount++;
    if (gap > stallLongestMs) stallLongestMs = gap;
    Serial.printf("Loop stalled %lu ms (Bluetooth blocked) - stall #%lu\n",
                  (unsigned long)gap, (unsigned long)stallCount);
  }
  if (safetyStopped) {
    safetyStopped = false;
    Serial.println("Command timeout - motors stopped by safety task, link kept");
  }

  readCommands();
  now = millis();

  // Connection change: always stop, reset the parser, keep accepting reconnects
  bool client = SerialBT.hasClient();
  if (client != hadClient) {
    hadClient = client;
    stopAll();
    lineLen  = 0;
    overflow = false;
    lastHbMs = now;
    xQueueReset(txQueue);              // nothing stale for a new client
    if (client) sendHeartbeat();       // tell the PC at once if we just rebooted
    Serial.println(client ? "Client connected" : "Client disconnected - stopped");
  }

  // (command watchdog: see safetyTask)
  applyMotors();                       // advance the soft start

  // Heartbeat: keeps the idle link active
  if (client && now - lastHbMs >= HEARTBEAT_MS) {
    lastHbMs = now;
    sendHeartbeat();
  }

  delay(1);                            // yield to the Bluetooth stack
}
