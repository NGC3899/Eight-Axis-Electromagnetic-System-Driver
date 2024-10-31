#!/bin/python3
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import *
import serial.tools
from tkinter import ttk
import time
import rospy
from std_msgs.msg import String
import numpy as np
import numpy
import math
from agent_new import AI_Agent
import calculation
import pdcontroller
import oritest

Pxz="(100,200)"
Pyz="(600,700)"

def callback(msg):

    global Pxz

    Pxz=msg.data

    #rospy.loginfo("XZ coordinate is:%s", Pxz)

def callback_2(msg):

    global Pyz

    Pyz=msg.data

    #rospy.loginfo("YZ corrdinate is:%s", Pyz)

def red_callback(data):
    global red_coord
    red_coord = data.data
    # rospy.loginfo(f"Received Red Coordinate: {red_coord}")

def blue_callback(data):
    global blue_coord
    blue_coord = data.data
    # rospy.loginfo(f"Received Blue Coordinate: {blue_coord}")

def red_callback_2(data):
    global red_coord_2
    red_coord_2 = data.data
    # rospy.loginfo(f"Received Red Coordinate: {red_coord}")

def blue_callback_2(data):
    global blue_coord_2
    blue_coord_2 = data.data
    # rospy.loginfo(f"Received Blue Coordinate: {blue_coord}")


def string_to_numpy_array(data):
        try:

            if isinstance(data,String):
                data_string=data.data
            elif isinstance(data,str):
                data_string=data
            else:
                raise ValueError("Input must be a string or std_msgs/String")

            data_string=data_string.replace('(','').replace(')','')
            coords=[float(coord.strip()) for coord in data_string.split(',')]
            if len(coords) !=2:
                raise ValueError("The input string does not contain exactly two coordinates")
            #coords.append(100.0)
            return np.array(coords)
        except ValueError as e:
            rospy.logerr("Error converting string to numpy array: %s",str(e))
            return None


def Open_System():

    # Step 1: Open 8 machines
    Remote_Write_Voltage("/dev/ttyUSB0",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB0",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB0",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB1",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB1",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB1",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB2",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB2",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB2",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB3",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB3",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB3",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB4",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB4",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB4",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB5",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB5",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB5",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB6",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB6",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB6",9600)
    time.sleep(0.01)

    Remote_Write_Voltage("/dev/ttyUSB7",9600,3000)
    time.sleep(0.01)
    Remote_Write_Current("/dev/ttyUSB7",9600,10)
    time.sleep(0.01)
    Remote_Open_Machine("/dev/ttyUSB7",9600)
    time.sleep(0.01)


def Coordinate_Transformation():

    global Pxz
    global Pyz
    global red_coord
    global blue_coord


    #rospy.init_node("Pn_Publisher",anonymous=True)
    #pub=rospy.Publisher('localframe', String,queue_size=1)
    #rate=rospy.Rate(20)
    #print("Pxz:",Pxz)
    #print("Pyz:",Pyz)
    Pxz_numpy=string_to_numpy_array(Pxz)
    Pyz_numpy=string_to_numpy_array(Pyz)
    P_numpy=np.insert(Pxz_numpy,1,Pyz_numpy[0])
    P_numpy=np.append(P_numpy,1)
    rospy.loginfo("P is%s",P_numpy)
    rospy.loginfo("P shape is:%s",P_numpy.shape)
    P1=np.matmul(np.linalg.pinv(T1),P_numpy)
    P2=np.matmul(np.linalg.pinv(T2),P_numpy)
    P3=np.matmul(np.linalg.pinv(T3),P_numpy)
    P4=np.matmul(np.linalg.pinv(T4),P_numpy)
    P5=np.matmul(np.linalg.pinv(T5),P_numpy)
    P6=np.matmul(np.linalg.pinv(T6),P_numpy)
    P7=np.matmul(np.linalg.pinv(T7),P_numpy)
    P8=np.matmul(np.linalg.pinv(T8),P_numpy)

    Current_Value_1="1,0,0,0,0,0,0,0"
    Current_Value_2="0,1,0,0,0,0,0,0"
    Current_Value_3="0,0,1,0,0,0,0,0"
    Current_Value_4="0,0,0,1,0,0,0,0"
    Current_Value_5="0,0,0,0,1,0,0,0"
    Current_Value_6="0,0,0,0,0,1,0,0"
    Current_Value_7="0,0,0,0,0,0,1,0"
    Current_Value_8="0,0,0,0,0,0,0,1"

    AI_1="No.1-1A"
    AI_2="No.2-1A"
    AI_3="No.3-1A"
    AI_4="No.4-1A"
    AI_5="No.5-1A"
    AI_6="No.6-1A"
    AI_7="No.7-1A"
    AI_8="No.8-1A"

    P1_str=f"{str(P1[0])},{str(P1[1])},{str(P1[2])}"
    P2_str=f"{str(P2[0])},{str(P2[1])},{str(P2[2])}"
    P3_str=f"{str(P3[0])},{str(P3[1])},{str(P3[2])}"
    P4_str=f"{str(P4[0])},{str(P4[1])},{str(P4[2])}"
    P5_str=f"{str(P5[0])},{str(P5[1])},{str(P5[2])}"
    P6_str=f"{str(P6[0])},{str(P6[1])},{str(P6[2])}"
    P7_str=f"{str(P7[0])},{str(P7[1])},{str(P7[2])}"
    P8_str=f"{str(P8[0])},{str(P8[1])},{str(P8[2])}"

    P1_input=','.join([AI_1,P1_str,Current_Value_1])
    print(P1_input)
    agent_1=AI_Agent(P1_input)
    result_1=agent_1.process_and_predict()
    # result_1=np.split(result_1,2)
    Predicted_Value_1=np.array([result_1[0], result_1[1], result_1[2]])
    Gradient_1=np.array([result_1[3], result_1[4], result_1[5], result_1[6], result_1[7], result_1[8], result_1[9], result_1[10], result_1[11]])

    P2_input=','.join([AI_2,P2_str,Current_Value_2])
    print(P2_input)
    agent_2=AI_Agent(P2_input)
    result_2=agent_2.process_and_predict()
    # result_2=np.split(result_2,2)
    Predicted_Value_2=np.array([result_2[0], result_2[1], result_2[2]])
    Gradient_2=np.array([result_2[3], result_2[4], result_2[5], result_2[6], result_2[7], result_2[8], result_2[9], result_2[10], result_2[11]])

    P3_input=','.join([AI_3,P3_str,Current_Value_3])
    print(P3_input)
    agent_3=AI_Agent(P3_input)
    result_3=agent_3.process_and_predict()
    # result_3=np.split(result_3,2)
    Predicted_Value_3=np.array([result_3[0], result_3[1], result_3[2]])
    Gradient_3=np.array([result_3[3], result_3[4], result_3[5], result_3[6], result_3[7], result_3[8], result_3[9], result_3[10], result_3[11]])
    
    P4_input=','.join([AI_4,P4_str,Current_Value_4])
    print(P4_input)
    agent_4=AI_Agent(P4_input)
    result_4=agent_4.process_and_predict()
    # result_4=np.split(result_4,2)
    Predicted_Value_4=np.array([result_4[0], result_4[1], result_4[2]])
    Gradient_4=np.array([result_4[3], result_4[4], result_4[5], result_4[6], result_4[7], result_4[8], result_4[9], result_4[10], result_4[11]])

    P5_input=','.join([AI_5,P5_str,Current_Value_5])
    print(P5_input)
    agent_5=AI_Agent(P5_input)
    result_5=agent_5.process_and_predict()
    # result_5=np.split(result_5,2)
    Predicted_Value_5=np.array([result_5[0], result_5[1], result_5[2]])
    Gradient_5=np.array([result_5[3], result_5[4], result_5[5], result_5[6], result_5[7], result_5[8], result_5[9], result_5[10], result_5[11]])

    P6_input=','.join([AI_6,P6_str,Current_Value_6])
    print(P6_input)
    agent_6=AI_Agent(P6_input)
    result_6=agent_6.process_and_predict()
    # result_6=np.split(result_6,2)
    Predicted_Value_6=np.array([result_6[0], result_6[1], result_6[2]])
    Gradient_6=np.array([result_6[3], result_6[4], result_6[5], result_6[6], result_6[7], result_6[8], result_6[9], result_6[10], result_6[11]])

    P7_input=','.join([AI_7,P7_str,Current_Value_7])
    print(P7_input)
    agent_7=AI_Agent(P7_input)
    result_7=agent_7.process_and_predict()
    # result_7=np.split(result_7,2)
    Predicted_Value_7=np.array([result_7[0], result_7[1], result_7[2]])
    Gradient_7=np.array([result_7[3], result_7[4], result_7[5], result_7[6], result_7[7], result_7[8], result_7[9], result_7[10], result_7[11]])

    P8_input=','.join([AI_8,P8_str,Current_Value_8])
    print(P8_input)
    agent_8=AI_Agent(P8_input)
    result_8=agent_8.process_and_predict()
    # result_8=np.split(result_8,2)
    Predicted_Value_8=np.array([result_8[0], result_8[1], result_8[2]])
    Gradient_8=np.array([result_8[3], result_8[4], result_8[5], result_8[6], result_8[7], result_8[8], result_8[9], result_8[10], result_8[11]])

    #print("No.1 is:",result_1)
    #P_str=f"{P1_str} {P2_str} {P3_str} {P4_str} {P5_str} {P6_str} {P7_str} {P8_str}"
    #pub.publish(P_str)
    #rospy.loginfo("The Position of the tips in each coordinate should be:%s",P_str)
    #rate.sleep()

    # PD Controller
    results = [result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8]

    red_1 = string_to_numpy_array(red_coord)
    #red_1 = string_to_numpy_array(red1)
    blue_1 = string_to_numpy_array(blue_coord)
    #blue_1 = string_to_numpy_array(blue1)
    red_2 = string_to_numpy_array(red_coord_2)
    #red_1 = string_to_numpy_array(red1)
    blue_2 = string_to_numpy_array(blue_coord_2)
    #blue_1 = string_to_numpy_array(blue1)
    
    M_vecter = oritest.calculate_v()
    M_norm = 1
    M = [M1 * M_norm for M1 in M_vecter]
    A = calculation.Calculation_A(M, results)
    kp = 0.07*np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(6,3)
    kd = 0.05*np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(6,3)
    dt = 0.01
    desired_P = np.array([10,100,50]).reshape(3,1)
    current_P = P_numpy[:3].reshape(3, 1)
    controller = pdcontroller.PDController(A, kp, kd, desired_P, current_P)
    # while True:
    TF = controller.Calculation_TF(desired_P, current_P, dt)
    if all(abs(P1 - P2) <= 1 for P1, P2 in zip(desired_P, current_P)):
        rospy.signal_shutdown('Target position reached')
    else:
        current = controller.Calculation_Current(TF)
        current = current * 100
        current = [abs(int(i)) for i in current] 
        print("Current is:",current) 
        
    if current[0]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB0",9600,current[0])
        time.sleep(0.01)

    if current[1]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB1",9600,current[1])
        time.sleep(0.01)
    
    if current[2]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB2",9600,current[2])
        time.sleep(0.01)

    if current[3]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB3",9600,current[3])
        time.sleep(0.01)

    if current[4]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB4",9600,current[4])
        time.sleep(0.01)

    if current[5]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB5",9600,current[5])
        time.sleep(0.01)

    if current[6]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB6",9600,current[6])
        time.sleep(0.01)

    if current[7]>300:
        pass
    else:
        Remote_Write_Current("/dev/ttyUSB7",9600,current[7])
        time.sleep(0.01)
        
a = 51/(2**(1/2))
theta_1 = math.pi
theta_2 = 3*math.pi/2
theta_4 = math.pi/2
theta_5 = 5*math.pi/4
theta_6 = 7*math.pi/4
theta_7 = math.pi/4
theta_8 = 3*math.pi/4
theta_down = 3*math.pi/4
theta_up = math.pi/4
T_down = np.array([[1,0,0,0],[0,math.cos(theta_down),-math.sin(theta_down),-a],[0,math.sin(theta_down),math.cos(theta_down),-a],[0,0,0,1]])
T_up = np.array([[1,0,0,0],[0,math.cos(theta_up),-math.sin(theta_up),-a],[0,math.sin(theta_up),math.cos(theta_up),a],[0,0,0,1]])

T1 = np.array([[math.cos(theta_1),-math.sin(theta_1),0,0],[math.sin(theta_1),math.cos(theta_1),0,0],[0,0,1,0],[0,0,0,1]]) @ T_down
T1_inverse = np.linalg.inv(T1)

T2 = np.array([[math.cos(theta_2),-math.sin(theta_2),0,0],[math.sin(theta_2),math.cos(theta_2),0,0],[0,0,1,0],[0,0,0,1]]) @ T_down
T2_inverse = np.linalg.inv(T2)

T3 = T_down
T3_inverse = np.linalg.inv(T3)

T4 = np.array([[math.cos(theta_4),-math.sin(theta_4),0,0],[math.sin(theta_4),math.cos(theta_4),0,0],[0,0,1,0],[0,0,0,1]]) @ T_down
T4_inverse = np.linalg.inv(T4)

T5 = np.array([[math.cos(theta_5),-math.sin(theta_5),0,0],[math.sin(theta_5),math.cos(theta_5),0,0],[0,0,1,0],[0,0,0,1]]) @ T_up
T5_inverse = np.linalg.inv(T5)

T6 = np.array([[math.cos(theta_6),-math.sin(theta_6),0,0],[math.sin(theta_6),math.cos(theta_6),0,0],[0,0,1,0],[0,0,0,1]]) @ T_up
T6_inverse = np.linalg.inv(T6)

T7 = np.array([[math.cos(theta_7),-math.sin(theta_7),0,0],[math.sin(theta_7),math.cos(theta_7),0,0],[0,0,1,0],[0,0,0,1]]) @ T_up
T7_inverse = np.linalg.inv(T7)

T8 = np.array([[math.cos(theta_8),-math.sin(theta_8),0,0],[math.sin(theta_8),math.cos(theta_8),0,0],[0,0,1,0],[0,0,0,1]]) @ T_up
T8_inverse = np.linalg.inv(T8)


aucCRCHi = [
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40
]

aucCRCLo = [
    0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06, 0x07, 0xC7,
    0x05, 0xC5, 0xC4, 0x04, 0xCC, 0x0C, 0x0D, 0xCD, 0x0F, 0xCF, 0xCE, 0x0E,
    0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09, 0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9,
    0x1B, 0xDB, 0xDA, 0x1A, 0x1E, 0xDE, 0xDF, 0x1F, 0xDD, 0x1D, 0x1C, 0xDC,
    0x14, 0xD4, 0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
    0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3, 0xF2, 0x32,
    0x36, 0xF6, 0xF7, 0x37, 0xF5, 0x35, 0x34, 0xF4, 0x3C, 0xFC, 0xFD, 0x3D,
    0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A, 0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38,
    0x28, 0xE8, 0xE9, 0x29, 0xEB, 0x2B, 0x2A, 0xEA, 0xEE, 0x2E, 0x2F, 0xEF,
    0x2D, 0xED, 0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
    0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60, 0x61, 0xA1,
    0x63, 0xA3, 0xA2, 0x62, 0x66, 0xA6, 0xA7, 0x67, 0xA5, 0x65, 0x64, 0xA4,
    0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F, 0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB,
    0x69, 0xA9, 0xA8, 0x68, 0x78, 0xB8, 0xB9, 0x79, 0xBB, 0x7B, 0x7A, 0xBA,
    0xBE, 0x7E, 0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
    0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71, 0x70, 0xB0,
    0x50, 0x90, 0x91, 0x51, 0x93, 0x53, 0x52, 0x92, 0x96, 0x56, 0x57, 0x97,
    0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C, 0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E,
    0x5A, 0x9A, 0x9B, 0x5B, 0x99, 0x59, 0x58, 0x98, 0x88, 0x48, 0x49, 0x89,
    0x4B, 0x8B, 0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
    0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42, 0x43, 0x83,
    0x41, 0x81, 0x80, 0x40
]

def crc16(frame, length):
    i = 0
    res = [0xFF, 0xFF]
    while length > 0:
        length -= 1
        iIndex = res[0] ^ frame[i]
        i += 1
        res[0] = res[1] ^ aucCRCHi[iIndex]
        res[1] = aucCRCLo[iIndex]
    return res

def split_hexadecimal(hex_number):
    if isinstance(hex_number, str):
        if hex_number.startswith("0x"):
            hex_number=int(hex_number,16)
        else:
            raise ValueError("String input must be a valid hexadecimal number starting with '0x")

    if not (0 <= hex_number <= 0xFFFF):
        raise ValueError("Input must be a 4-digit hexadecimal number within the range 0x0000 to 0xFFFF")
    
    high_byte=(hex_number & 0xFF00) >> 8
    low_byte = hex_number & 0x00FF
    return high_byte, low_byte

def Remote_Open_Machine(port:str,baudrate:int):
    
    SendCommand=[]
    SendCommand.append(0x01)
    SendCommand.append(0x10)
    SendCommand.append(0x00)
    SendCommand.append(0x42)
    SendCommand.append(0x00)
    SendCommand.append(0x02)
    SendCommand.append(0x04)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    SendCommand.append(0x01)                    
    #SendCommand=bytes([0x01,0x03,0x01,0x2D,0x00,0x01])
    #SendCommand=create_crc16_message(SendCommand)
    CRC=crc16(SendCommand,11)
    CRC_1=CRC[0]
    CRC_2=CRC[1]
    SendCommand.append(CRC_1)
    SendCommand.append(CRC_2)
    print("SendCommand:", SendCommand)
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            ser.write(SendCommand)
            #print("Data written to {port}: {data.hex()}")
    except serial.SerialException as e:
        print(f"Error: {e}")
    '''
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            while True:
                if ser.in_waiting>0:
                    data=ser.read(ser.in_waiting)
                    print(f"Received: {data.hex()}")
                    print(f"Received (ASCII): {data.decode(errors='ignore')}")

    except serial.SerialException as e:
        print(f"Error: {e}")
    '''
    
def Remote_Write_Voltage(port:str,baudrate:int, Voltage:int):
    
    SendCommand=[]
    SendCommand.append(0x01)
    SendCommand.append(0x10)
    SendCommand.append(0x00)
    SendCommand.append(0x40)
    SendCommand.append(0x00)
    SendCommand.append(0x02)
    SendCommand.append(0x04)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    Voltage_hex=hex(Voltage)
    Voltage_hex_1,Voltage_hex_2=split_hexadecimal(Voltage_hex)
    SendCommand.append(Voltage_hex_1)
    SendCommand.append(Voltage_hex_2)
    
    print("Hex Voltage:",Voltage_hex)
    print("Voltage_hex_1:",Voltage_hex_1)
    print("Voltage_hex_2:",Voltage_hex_2)
    #SendCommand=bytes([0x01,0x10,0x00,0x40,0x00,0x02,0x04,0x00,0x00,0x00,0x32])
    CRC=crc16(SendCommand,11)
    CRC_1=CRC[0]
    CRC_2=CRC[1]
    SendCommand.append(CRC_1)
    SendCommand.append(CRC_2)
    print("SendCommand:", SendCommand)
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            ser.write(SendCommand)
            #print(f"Data written to {port}: {data.hex()}")
    except serial.SerialException as e:
        print(f"Error: {e}")
'''
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            while True:
                if ser.in_waiting>0:
                    data=ser.read(ser.in_waiting)
                    print(f"Received: {data.hex()}")
                    print(f"Received (ASCII): {data.decode(errors='ignore')}")

    except serial.SerialException as e:
        print(f"Error: {e}")
'''    

def Remote_Write_Current(port:str,baudrate:int,Current:int):
    SendCommand=[]
    SendCommand.append(0x01)
    SendCommand.append(0x10)
    SendCommand.append(0x00)
    SendCommand.append(0x41)
    SendCommand.append(0x00)
    SendCommand.append(0x02)
    SendCommand.append(0x04)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    Current_hex=hex(Current)
    Current_hex_1,Current_hex_2=split_hexadecimal(Current_hex)
    SendCommand.append(Current_hex_1)
    SendCommand.append(Current_hex_2)
    CRC=crc16(SendCommand,11)
    CRC_1=CRC[0]
    CRC_2=CRC[1]
    SendCommand.append(CRC_1)
    SendCommand.append(CRC_2)
    print("SendCommand:", SendCommand)   
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            ser.write(SendCommand)
            #print("Data written to {port}: {data.hex()}")
    except serial.SerialException as e:
        print(f"Error: {e}")
    '''
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            while True:
                if ser.in_waiting>0:
                    data=ser.read(ser.in_waiting)
                    print(f"Received: {data.hex()}")
                    print(f"Received (ASCII): {data.decode(errors='ignore')}")

    except serial.SerialException as e:
        print(f"Error: {e}")
    '''

def Remote_Close_Machine(port:str,baudrate:int):
    SendCommand=[]
    SendCommand.append(0x01)
    SendCommand.append(0x10)
    SendCommand.append(0x00)
    SendCommand.append(0x42)
    SendCommand.append(0x00)
    SendCommand.append(0x02)
    SendCommand.append(0x04)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    SendCommand.append(0x00)
    CRC=crc16(SendCommand,11)
    CRC_1=CRC[0]
    CRC_2=CRC[1]
    SendCommand.append(CRC_1)
    SendCommand.append(CRC_2)
    print("SendCommand:", SendCommand)
    #SendCommand=create_crc16_message(SendCommand)
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            ser.write(SendCommand)
            #print("Data written to {port}: {data.hex()}")
    except serial.SerialException as e:
        print(f"Error: {e}")

    '''
    try:
        with serial.Serial(port,baudrate,timeout=1) as ser:
            while True:
                if ser.in_waiting>0:
                    data=ser.read(ser.in_waiting)
                    print(f"Received: {data.hex()}")
                    print(f"Received (ASCII): {data.decode(errors='ignore')}")

    except serial.SerialException as e:
        print(f"Error: {e}")
    '''

if __name__=="__main__":

    rospy.init_node("DecisionMaker",anonymous=True)
    Sub1=rospy.Subscriber("xzcoordinate",String,callback)
    Sub2=rospy.Subscriber("yzcoordinate",String,callback_2)
    red1 = rospy.Subscriber('red_coordinate', String, red_callback)
    blue1 = rospy.Subscriber('blue_coordinate', String, blue_callback)
    red2 = rospy.Subscriber('red_coordinate_2', String, red_callback_2)
    blue2 = rospy.Subscriber('blue_coordinate_2', String, blue_callback_2)
    rate=rospy.Rate(20)
    Open_System()
    while not rospy.is_shutdown():
        Coordinate_Transformation()
        rate.sleep()
