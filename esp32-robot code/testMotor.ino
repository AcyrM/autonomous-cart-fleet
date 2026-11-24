#include "BluetoothSerial.h"
#include <string.h>
#include <stdio.h>

#define pinMotor1_1 16     
#define pinMotor1_2 4      
#define pinMotor2_1 23     
#define pinMotor2_2 18   
#define pinMotor3_1 17   
#define pinMotor3_2 5  
#define pinMotor4_1 22    
#define pinMotor4_2 19

BluetoothSerial SerialBT;

int speed = 255;            // 255 = 46,12 cm/s;
char cmd = ' ';
int wait_time = 100, time_or_cmd, i;

///////////////////////////////////////////////////////////
//Classe do motor
class DCMotor {  
  int spd = 255, pin1, pin2;
  int spdstop = 0;
  public:  
  
    void Pinout(int in1, int in2){ // Method for declaring motor pinout
      pin1 = in1;
      pin2 = in2;
      pinMode(pin1, OUTPUT);
      pinMode(pin2, OUTPUT);
      }       void Speed(int in1){ // Method for saving motor speed
      spd = in1;
      }     
    void Forward(){ // Method to go forward
      analogWrite(pin1, spd);
      digitalWrite(pin2, LOW);
      }   
    void Backward(){ // Method to go backward
      digitalWrite(pin1, LOW);
      analogWrite(pin2, spd);
      }
    void Stop(){ // Method to stop
      analogWrite(pin1, spdstop);
      analogWrite(pin2, spdstop);
      }
   };

DCMotor Motor1, Motor2, Motor3, Motor4;                 // Motor objects 

//////////////////////////////////////////////////////////////
//Void Setup
void setup() {
  SerialBT.begin("Esp32-BT");
  Serial.begin(9600);
  Motor1.Pinout(pinMotor1_1,pinMotor1_2); // Set motor pins.
  Motor2.Pinout(pinMotor2_1,pinMotor2_2); 
  Motor3.Pinout(pinMotor3_1,pinMotor3_2);
  Motor4.Pinout(pinMotor4_1,pinMotor4_2);
}

///////////////////////////////////////////////////////////
//Void Loop
void loop() {
  if (SerialBT.available()){
    cmd = SerialBT.read();
  }
  if (cmd != ' '){
    motor_control(cmd, speed, wait_time, Motor1, Motor2, Motor3, Motor4);
  }
  cmd = ' ';
}

///////////////////////////////////////////////////////////
//Comandos de controle do Carrinho
void motor_control (char cmd, int speed,int del, DCMotor Motor1, DCMotor Motor2, DCMotor Motor3, DCMotor Motor4){
  if (cmd == 'F'){
    Motor1.Speed(speed); // Set motors speed
    Motor2.Speed(speed);
    Motor3.Speed(speed);
    Motor4.Speed(speed);
    Motor1.Forward(); 
    Motor2.Forward();
    Motor3.Forward();
    Motor4.Forward();
    Serial.print("FORWARD\n");
    delay(del);
  }
  else if (cmd == 'B'){
    Motor1.Speed(speed); // Set motors speed
    Motor2.Speed(speed);
    Motor3.Speed(speed);
    Motor4.Speed(speed);
    Motor1.Backward(); 
    Motor2.Backward();
    Motor3.Backward();
    Motor4.Backward();
    delay(del);
  }
  else if (cmd == 'L'){
    Motor1.Speed(speed); // Set motors speed
    Motor2.Speed(speed);
    Motor3.Speed(speed);
    Motor4.Speed(speed);
    Motor1.Forward(); 
    Motor2.Backward();
    Motor3.Backward();
    Motor4.Forward();
    delay(del);
  }
  else if (cmd == 'R'){
    Motor1.Speed(speed); // Set motors speed
    Motor2.Speed(speed);
    Motor3.Speed(speed);
    Motor4.Speed(speed);
    Motor1.Backward();
    Motor2.Forward();
    Motor3.Forward();
    Motor4.Backward();
    delay(del);
  }
  Serial.print("STOP\n");
  Motor1.Stop();
  Motor2.Stop();
  Motor3.Stop();
  Motor4.Stop();
}
