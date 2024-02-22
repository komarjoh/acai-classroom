# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tqdm import tqdm
import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import sys
import statistics
import datetime

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
#
# # Setting up the Pose function.
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
#
# # Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

## Define the areas of interest (e.g., teaching slides, students)
## This is to calculate total time of interaction with these areas of interest

# It has to be fed the video source as parameter, it then freezes the first frame of the video
video_name = 'test_HUGO.mp4'

# Create dictionary, one list for framenuber, one for teaching style (active or passive)
# Create dictionary, one count for frames spent near slides, one for near students area
teaching_style_dict = {'frame': [], 'teaching_style': [],
                       'slidesarea': [], 'studentsarea':[],
                       'pointingslides':[], 'lookingstudents':[],
                       'computerarea':[], 'whiteboardarea':[],
                       'facingstudents':[]}

# Create a dictionary for holding the landmark coordinates in CSV:
landmark_coordinate_dict = {}
for x in range(33):
    landmark_coordinate_dict[f"lm{x}_x"] = []
    landmark_coordinate_dict[f"lm{x}_y"] = []
    landmark_coordinate_dict[f"lm{x}_z"] = []
    landmark_coordinate_dict[f"lm{x}_visibility"] = []


def areasofinterest(video_name):
    drawing = False
    landmark_references = []

    aoi_count = 0

    # define mouse callback function to draw circle
    def draw_rectangle(event, x, y, flags, param):
        global drawing, temp_landmarks

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            temp_landmarks = []
            # Append origin
            temp_landmarks.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Append destination
            temp_landmarks.append((x, y))

            print(temp_landmarks)

            landmark_references.append(temp_landmarks)

    vid = cv2.VideoCapture(video_name)

    while(True):
        ret, frame = vid.read()
        cv2.namedWindow("Define Areas of Interest")

        cv2.setMouseCallback("Define Areas of Interest", draw_rectangle)



        if len(landmark_references) == 0:
            cv2.putText(frame, f"Mark out teaching slides", (10,30),
                        cv2.QT_FONT_NORMAL, 0.8, (255,255,255), 1)

        if len(landmark_references) == 1:
            cv2.putText(frame, f"Mark out area where students are sitting", (10,30),
                        cv2.QT_FONT_NORMAL, 0.8, (255,255,255), 1)

        if len(landmark_references) == 2:
            cv2.putText(frame, f"Mark out area where computer/laptop is", (10,30),
                        cv2.QT_FONT_NORMAL, 0.8, (255,255,255), 1)

        if len(landmark_references) >2:
            cv2.putText(frame, f"Mark out area(s) whiteboards are", (10,30),
                        cv2.QT_FONT_NORMAL, 0.8, (255,255,255), 1)

        # Draw rectangle based on bounding boxes
        # aoi_bool = 0

        # label_index = 0
        # for lm in areasofinterest_list:
        #     if label_index > 3:
        #         label = 'Whiteboard'
        #     else:
        #         label = aoilabels_list[label_index]
        #
        #     overlay = frame.copy()
        #     alpha = 0.85
        #     cv2.rectangle(frame, lm[0], lm[1], (0, 0, 255), -1)
        #     cv2.putText(frame, f"{label}", (int((lm[1][0] + lm[0][0]) / 2), int((lm[1][1] + lm[0][1]) / 2)),
        #                 cv2.QT_FONT_NORMAL, 0.8, (255, 255, 255), 1)
        #
        #     frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        #
        #     label_index += 1
        aoilabels_list = ['Teaching Slides', 'Students', 'Computer', 'Whiteboard']
        label_index = 0
        for lm in landmark_references:
            if label_index > 2:
                label = 'Whiteboard'
            else:
                label = aoilabels_list[label_index]

            overlay = frame.copy()
            alpha = 0.5
            cv2.rectangle(frame, lm[0], lm[1], (0, 0, 255), -1)
            cv2.putText(frame, f"{label}", (int((lm[1][0]+lm[0][0])/2), int((lm[1][1]+lm[0][1])/2)),
                        cv2.QT_FONT_NORMAL, 0.4, (255,255,255), 1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            label_index+=1

            # Display the resulting frame
        cv2.imshow('Define Areas of Interest', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    # We should return the list of landmark references
    return landmark_references

# Run function, if necessary
areasofinterest_list = areasofinterest(video_name)
print(areasofinterest_list)

# While testing, pre-define AOI
'''areasofinterest_list = [[(1286, 73), (2066, 577)],
                        [(0, 1000), (3365, 3200)],
                        [(255, 538), (648, 769)],
                        [(618, 251), (1213, 589)]]
'''
# Create pose detection function
def detectPose(image, pose):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))


    # out.write(output_image)
    return output_image, landmarks, results

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle

def classifyPose(landmarks, output_image, history, current_frame_count):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''

    # Initialize the label of the pose. It is not known at this stage.
    label = 'Passive Teaching'
    # print(label, output_image)
    color = (0, 0, 255)

    # Create a variable to manage the logging of facing forwards variable
    facingstudentslog = False ### If true, means it has already been logged

    history_Rwrist_x = []
    history_Rwrist_y = []
    history_Lwrist_x = []
    history_Lwrist_y = []
    for lm in history:
        history_Rwrist_x.append(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value][0])
        history_Rwrist_y.append(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value][1])

        history_Lwrist_x.append(lm[mp_pose.PoseLandmark.LEFT_WRIST.value][0])
        history_Lwrist_y.append(lm[mp_pose.PoseLandmark.LEFT_WRIST.value][1])

    # If at least 30 frames of history detected, then run wrist classficiation
    if len(history) >= 30:
        # print(len(history_Rwrist_x))
        # Check right wrist deviation
        # print('right wrist deviation from 60 frames:')
        # print(statistics.stdev(history_Rwrist_x))
        #
        # print('left wrist deviation from 60 frames:')
        # print(statistics.stdev(history_Lwrist_x))
        Rwrist_devation_x = int(statistics.stdev(history_Rwrist_x))
        Lwrist_devation_x = int(statistics.stdev(history_Lwrist_x))

        # For wrist functions
        # cv2.putText(output_image, f"Right wrist deviation: {Rwrist_devation_x}", (10, 50),
        #             cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        # cv2.putText(output_image, f"Left wrist deviation: {Lwrist_devation_x}", (10, 60),
        #             cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        # If deviation is high, threshold set as 10, change label of classified pose
        if Rwrist_devation_x > 10 and Lwrist_devation_x > 10:
            label = 'Active Gesturing'

    ############################################################################################
    # Look at where the lecturer is pointing towards, if towards slides then count frames     #
    ############################################################################################
    left_shoulder_lcn = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_wrist_lcn = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder_lcn = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_wrist_lcn = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # left_pointing_angle = math.atan((left_shoulder_lcn[1] - left_wrist_lcn[1]) / (left_wrist_lcn[0] - left_shoulder_lcn[0]))
    # cv2.putText(output_image, str(round(left_pointing_angle, 1)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    ## Create two lists to hold the pointing coordinates
    left_line_pointing_list = []
    right_line_pointing_list = []

    ###############################
    # THIS BLOCK IS FOR LEFT HAND #
    ###############################
    # print("x wrist", left_wrist_lcn[0], "x shoulder",left_shoulder_lcn[0])
    # print("y wrist", left_wrist_lcn[1], "y shoulder",left_shoulder_lcn[1])

    spacetillendwidth = frame_width - left_wrist_lcn[0]
    ## Assuming space for one tick is 10px
    tickstillendwidth = int(spacetillendwidth / 200)
    # print('number of ticks = ', tickstillendwidth)

    if left_wrist_lcn[1] > areasofinterest_list[0][1][1]:
        # cv2.putText(output_image, "Left wrist below lowest point of slides", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2) ## This means its impossible that lecturer is pointing to slides
        print("Left wrist below lowest point of slides")

    else:
        # cv2.putText(output_image, "Left wrist ABOVE lowest point of slides", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        print("Left wrist above lowest point of slides")

        try:
            left_line_pointing_list = []
            # For left arm, if wrist is to the left of the shoulder (most likely not pointing to slides, so break loop)
            if left_wrist_lcn[0] < left_shoulder_lcn[0]:
                pass
            else:
                # If wrist is lower than shoulder (i.e., y coordinate), calculate differently
                if left_wrist_lcn[1] > left_shoulder_lcn[1]:
                    print('Wrist below shoulder')
                    left_pointing_angle = math.degrees(math.atan((left_wrist_lcn[1] - left_shoulder_lcn[1]) / (left_wrist_lcn[0] - left_shoulder_lcn[0])))
                    # cv2.putText(output_image, "Wrist below shoulder", (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                    # cv2.putText(output_image, str(round(left_pointing_angle, 1)), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                else:
                    print('Wrist above shoulder')
                    left_pointing_angle = math.degrees(math.atan((left_shoulder_lcn[1] - left_wrist_lcn[1]) / (left_wrist_lcn[0] - left_shoulder_lcn[0])))
                    # cv2.putText(output_image, "Wrist above shoulder", (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                    # cv2.putText(output_image, str(round(left_pointing_angle, 1)), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                    for x in range(tickstillendwidth):
                        # Extended x = wrist + 10
                        pixelswithinline = 100
                        # print('small triangle angle = ', left_pointing_angle)
                        extendedx = (left_wrist_lcn[0] - left_shoulder_lcn[0]) + (pixelswithinline * x)
                        extendedy = extendedx * math.tan(left_pointing_angle * math.pi / 180)
                        # print('small triangle y =', extendedy)

                        newxcoord = left_wrist_lcn[0] + (pixelswithinline * x)
                        newycoord = left_shoulder_lcn[1] - extendedy

                        # print(extendedx, newycoord)

                        left_line_pointing_list.append((int(newxcoord), int(newycoord)))

                    # print(left_line_pointing_list)
                    print('printing line with left wrist above shoulder')
                    for values in left_line_pointing_list:
                        cv2.circle(output_image, (values[0], values[1]), 5, (0, 255, 255), 5)

        except Exception as e:
            print('Error with calculating pointing angle for left hand')
            print(e)

    ################################
    # THIS BLOCK IS FOR RIGHT HAND #
    ################################
    # print('Right hand:')
    # print("x wrist",  right_wrist_lcn[0], "x shoulder", right_shoulder_lcn[0])
    # print("y wrist",  right_wrist_lcn[1], "y shoulder", right_shoulder_lcn[1])

    spacetillendwidth = frame_width - right_wrist_lcn[0]
    ## Assuming space for one tick is 10px
    tickstillendwidth = int(spacetillendwidth / 200)
    # print('number of ticks = ', tickstillendwidth)

    if right_wrist_lcn[1] > areasofinterest_list[0][1][1]:
        # cv2.putText(output_image, "Right wrist below lowest point of slides", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5,
        #             color, 2)  ## This means its impossible that lecturer is pointing to slides
        print("Right wrist below lowest point of slides")

    else:
        cv2.putText(output_image, "Right wrist ABOVE lowest point of slides", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    color, 2)

        try:
            right_line_pointing_list = []
            # For right arm, if wrist is to the left of the shoulder (most likely not pointing to slides, so break loop)
            if right_wrist_lcn[0] > right_shoulder_lcn[0]:
                # Eventually add in something here to account for movement of lecturer
                print("When facing forward, wrist is to the right of the shoulder, doesn't seem to be pointing at slides")
                pass
            else:
                # If wrist is lower than shoulder (i.e., y coordinate), calculate differently
                if right_wrist_lcn[1] > right_shoulder_lcn[1]:
                    print('Wrist below shoulder')
                    right_pointing_angle = math.degrees(math.atan((right_wrist_lcn[1] - right_shoulder_lcn[1]) / (right_wrist_lcn[0] - right_shoulder_lcn[0])))
                    cv2.putText(output_image, "Wrist below shoulder", (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                    cv2.putText(output_image, str(round(right_pointing_angle, 1)), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                                color, 2)

                else:
                    print('Wrist above shoulder')
                    right_pointing_angle = math.degrees(math.atan((right_wrist_lcn[1] - right_shoulder_lcn[1]) / (right_wrist_lcn[0] - right_shoulder_lcn[0])))
                    cv2.putText(output_image, "Wrist above shoulder", (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                    cv2.putText(output_image, str(round(right_pointing_angle, 1)), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                                color, 2)

                    for x in range(tickstillendwidth):
                        # Extended x = wrist + 10
                        pixelswithinline = 100
                        # print('small triangle angle = ', right_pointing_angle)
                        extendedx = (right_shoulder_lcn[0] - right_wrist_lcn[0]) + (pixelswithinline * x)
                        extendedy = extendedx * math.tan(right_pointing_angle * math.pi / 180)
                        # print('small triangle y =', extendedy)

                        newxcoord = right_wrist_lcn[0] - (pixelswithinline * x)
                        newycoord = right_shoulder_lcn[1] - extendedy

                        # print(extendedx, newycoord)

                        right_line_pointing_list.append((int(newxcoord), int(newycoord)))

                    # print(right_line_pointing_list)
                    print('printing line with right wrist above shoulder')
                    for values in right_line_pointing_list:
                        cv2.circle(output_image, (values[0], values[1]), 5, (0, 255, 255), 5)

        except Exception as e:
            print('Error with calculating pointing angle for right hand')
            print(e)

    checkifpointing = False
    for coordinates in left_line_pointing_list:
        # Check that pointing dots are within the area of the slides
        # x value of coordinate
        if areasofinterest_list[0][0][0] <= coordinates[0] <= areasofinterest_list[0][1][0]:
            if areasofinterest_list[0][0][1] <= coordinates[1] <= areasofinterest_list[0][1][1]:
                # line points within slides
                # both x and y coordinates are within slide area
                print('lecturer pointing at slides')
                # cv2.putText(output_image, 'Lecturer pointing at slides', (10, 250), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                checkifpointing = True
                break

    # If checkifpoint is False even after checking left hand, then check if right hand is pointing at slides
    if checkifpointing == False:
        for coordinates in right_line_pointing_list:
            if areasofinterest_list[0][0][0] <= coordinates[0] <= areasofinterest_list[0][1][0]:
                if areasofinterest_list[0][0][1] <= coordinates[1] <= areasofinterest_list[0][1][1]:
                    print('lecturer pointing at slides')
                    cv2.putText(output_image, 'Lecturer pointing at slides', (10, 250), cv2.FONT_HERSHEY_PLAIN, 2,
                                color, 2)

                    checkifpointing = True


    if checkifpointing == True:
        teaching_style_dict['pointingslides'].append(1)
    else:
        teaching_style_dict['pointingslides'].append(0)


    ################################################################################################
    #                           CHECK THE LINE OF SIGHT OF LECTURER                                #
    #          Idea is form a triangle between center of head, and R shoulder + L shoulder         #
    ################################################################################################


    # If xvalue of R shoulder less than L shoulder = lecturer not facing front
    # Then don't bother with eyelines -> not looking at class
    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0]:
        cv2.putText(output_image, "Lecturer not facing forwards", (10, 500),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

        if facingstudentslog == False:
            teaching_style_dict['facingstudents'].append(0)
            facingstudentslog = True

        #######################################################################################
        #   If lecturer not facing the class - check if he's near computer or whiteboard      #
        #######################################################################################
        shoulders_mid = (int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] +
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2),
                    int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] +
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2))\

        ## Check if whiteboards are defined, if run smoothly the whiteboard should be defined as AOI 3 and onwards
        facingaway_nearwhiteboard = False
        if len(areasofinterest_list) > 2:
            for temp_aoi in areasofinterest_list[3:]:
                if temp_aoi[0][0] <= shoulders_mid[0] <= temp_aoi[1][0] and temp_aoi[0][1] <= shoulders_mid[1] <= temp_aoi[1][1]:
                    facingaway_nearwhiteboard = True

        if facingaway_nearwhiteboard == True:
            teaching_style_dict['whiteboardarea'].append(1)
            cv2.putText(output_image, "Facing away & Near Whiteboard", (10, 750), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

        else:
            teaching_style_dict['whiteboardarea'].append(0)


    else:
        lineofsight_students = False
        if landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][0] > landmarks[mp_pose.PoseLandmark.NOSE.value][0]:
            ## Lecturer is facing to the left side of screen
            ## For left ear to nose
            ## Calculate point where ear - nose - connects to frame of video
            try:
                y_certainpoint = ((landmarks[mp_pose.PoseLandmark.NOSE.value][1] - landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][1]) / (landmarks[mp_pose.PoseLandmark.NOSE.value][0] - landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][0])) * (0-landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][0]) + landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][1]
                y_certainpoint = int(y_certainpoint)
                print(y_certainpoint)
            except Exception as e:
                print(e)
                ## Sometimes there's division by zero error if the values are similar
                ## In this case force y_certainpoint as frame height
                y_certainpoint = frame_height

            if y_certainpoint < frame_height:
                # cv2.putText(output_image, "Intersect within frame", (10, 1800), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                cv2.line(output_image, (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][0], landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][1]), (0, int(y_certainpoint)), (255,255,255))

                if y_certainpoint > areasofinterest_list[1][0][1]:
                    cv2.putText(output_image, "Line of sight to students", (10, 1500), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 0, 0),
                                2)
                    lineofsight_students = True
                else:
                    cv2.putText(output_image, "Line of sight NOT to students", (10, 1500), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 0, 0),
                                2)
            else:
                # cv2.putText(output_image, "Intersect outside frame", (10, 1800), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                pass
        if landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][0] < landmarks[mp_pose.PoseLandmark.NOSE.value][0]:
            ## Lecturer is facing to the right side of screen
            ## For right ear to nose
            ## Calculate point where ear - nose - connects to frame of video
            try:
                y_certainpoint = ((landmarks[mp_pose.PoseLandmark.NOSE.value][1] -
                                   landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][1]) / (
                                              landmarks[mp_pose.PoseLandmark.NOSE.value][0] -
                                              landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][0])) * (
                                             frame_width - landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][0]) + \
                                 landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][1]
                y_certainpoint = int(y_certainpoint)
                print(y_certainpoint)

            except Exception as e:
                print(e)
                ## Sometimes there's division by zero error if the values are similar
                ## In this case force y_certainpoint as frame height
                y_certainpoint = frame_height

            if y_certainpoint < frame_height:
                # cv2.putText(output_image, "Intersect within frame", (10, 1800), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                #             2)
                cv2.line(output_image, (landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][0], landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][1]),
                         (frame_width, int(y_certainpoint)), (255, 255, 255))

                if y_certainpoint > areasofinterest_list[1][0][1]:
                    cv2.putText(output_image, "Line of sight to students", (10, 1650), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 0, 0),
                                2)
                    lineofsight_students = True
                else:
                    cv2.putText(output_image, "Line of sight NOT to students", (10, 1650), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 0, 0),
                                2)
            else:
                # cv2.putText(output_image, "Intersect outside frame", (10, 1800), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                #             2)
                pass


        if lineofsight_students == True:
            teaching_style_dict['lookingstudents'].append(1)
        else:
            teaching_style_dict['lookingstudents'].append(0)


        ### Check of both eyes are visible
        # print(landmarks[mp_pose.PoseLandmark.])
        print(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value])
        print(landmarks[mp_pose.PoseLandmark.RIGHT_EAR])

        ## Calculate middle of two eyes = HEAD
        head_mid = (int((landmarks[mp_pose.PoseLandmark.LEFT_EYE.value][0] + landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value][0]) / 2),
                    int((landmarks[mp_pose.PoseLandmark.LEFT_EYE.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value][1]) / 2))

        try:
            angle_head_Rshoulder = math.degrees(math.atan((head_mid[0] - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1] - head_mid[1])))
        except Exception as e:
            print(e)
            angle_head_Rshoulder = 1

        try:
            angle_head_Lshoulder = math.degrees(math.atan((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] - head_mid[0]) / (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - head_mid[1])))
        # print('right shoulder angle = ', angle_head_Rshoulder, 'left shoulder angle = ', angle_head_Lshoulder,)
        except Exception as e:
            print(e)
            angle_head_Lshoulder = 1

        Rshoulder_to_frame = (frame_height - head_mid[1]) * math.tan(angle_head_Rshoulder * math.pi / 180)
        Lshoulder_to_frame = (frame_height - head_mid[1]) * math.tan(angle_head_Lshoulder * math.pi / 180)

        # print('right shoulder angle = ', Rshoulder_to_frame)

        xvalue_Rshoulder_toframe = int(head_mid[0] - Rshoulder_to_frame)
        xvalue_Lshoulder_toframe = int(head_mid[0] + Lshoulder_to_frame)

        cv2.line(output_image, head_mid, (xvalue_Rshoulder_toframe, frame_height), (0,255,0), 1)
        cv2.line(output_image, head_mid, (xvalue_Lshoulder_toframe, frame_height), (0, 255, 0), 1)

        #####################################
        ### APPEND TO TEACHING STYLE DICT ###
        #####################################
        # Check 2nd item in aoi list -> students area
        if areasofinterest_list[1][0][0] <= xvalue_Rshoulder_toframe or xvalue_Lshoulder_toframe < areasofinterest_list[1][1][0]:
            cv2.putText(output_image, "Body facing students", (10, 1350), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            if facingstudentslog == False:
                teaching_style_dict['facingstudents'].append(1)
                facingstudentslog = True


    # Write the label on the output image.
    if label == 'Passive Teaching':
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    teaching_style_dict['frame'].append(current_frame_count)
    teaching_style_dict['teaching_style'].append(label)

    return output_image, label, teaching_style_dict

def trackCOG(landmarks, output_image, aoi_list):
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]

    ## Calculate middle of two eyes = HEAD
    head_mid = (int((landmarks[mp_pose.PoseLandmark.LEFT_EYE.value][0] + landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value][0]) / 2),
                int((landmarks[mp_pose.PoseLandmark.LEFT_EYE.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value][1]) / 2))

    cv2.circle(output_image, head_mid, 5, (0, 255, 255), 5)

    if left_heel and right_heel:
        # print(left_heel, right_heel)
        center_of_gravity = ((left_heel[0] + right_heel[0]) / 2, (left_heel[1] + right_heel[1]) / 2, (left_heel[2] + right_heel[2]) / 2)
        # print("cg = ", center_of_gravity)
        cv2.circle(output_image, (int(center_of_gravity[0]), int(center_of_gravity[1])), 5, (255, 0, 0), 5)
        list_of_COG.append((int(center_of_gravity[0]), int(center_of_gravity[1])))
        # print((int(center_of_gravity[0]), int(center_of_gravity[1])))

    ## If head near slides then count frame // if COG near students then count frame
    # Double check that there are AOIs, if none then move on
    if len(aoi_list) > 0:
        slides_bool, students_bool = False, False

        ## If x & y coord of head within x & y coordinates of slides, then
        if (aoi_list[0][0][0] <= head_mid[0] <= aoi_list[0][1][0]) and (aoi_list[0][0][1] <= head_mid[1] <= aoi_list[0][1][1]):
            slides_bool = True

        if (aoi_list[1][0][0] <= center_of_gravity[0] <= aoi_list[1][1][0]) and (aoi_list[1][0][1] <= center_of_gravity[1] <= aoi_list[1][1][1]):
            students_bool = True

        if slides_bool == False and students_bool == False:
            teaching_style_dict['slidesarea'].append(0)
            teaching_style_dict['studentsarea'].append(0)

        if slides_bool == True and students_bool == True:
            cv2.putText(output_image, 'Interacting with slides', (10, 90), cv2.QT_FONT_NORMAL, 1, (1, 194, 252), 2)
            cv2.putText(output_image, 'Interacting with students', (10, 110), cv2.QT_FONT_NORMAL, 1, (0,254,255), 2)
            teaching_style_dict['slidesarea'].append(1)
            teaching_style_dict['studentsarea'].append(1)

        elif slides_bool == True:
            cv2.putText(output_image, 'Interacting with slides', (10, 90), cv2.QT_FONT_NORMAL, 1, (1, 194, 252), 2)
            teaching_style_dict['slidesarea'].append(1)
            teaching_style_dict['studentsarea'].append(0)

        elif students_bool == True:
            cv2.putText(output_image, 'Interacting with students', (10, 90), cv2.QT_FONT_NORMAL, 1, (0,254,255), 2)
            teaching_style_dict['studentsarea'].append(1)
            teaching_style_dict['slidesarea'].append(0)


        ################################################################
        # Check if lecturer - shoulders are near computer              #
        ################################################################
        shoulders_mid = (int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] +
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2),
                    int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] +
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2))

        cv2.circle(output_image, shoulders_mid, 5, (102, 5, 255), 2)

        # If middle of shoulders within rectangle box for computer, then count
        if aoi_list[2][0][0] < shoulders_mid[0] < aoi_list[2][1][0] and aoi_list[2][0][1] < shoulders_mid[1] < aoi_list[2][1][1]:
            cv2.putText(output_image, 'Lecturer near computer', (10, 900), cv2.QT_FONT_NORMAL, 1, (102, 5, 255), 2)
            teaching_style_dict['computerarea'].append(1)
        else:
            teaching_style_dict['computerarea'].append(0)


        ################################################################
        # Check if lecturer - shoulders are near whiteboard              #
        ################################################################
        # If middle of shoulders within rectangle box for computer, then count
        if aoi_list[3][0][0] < shoulders_mid[0] < aoi_list[3][1][0] and aoi_list[3][0][1] < shoulders_mid[1] < aoi_list[3][1][1]:
            cv2.putText(output_image, 'Lecturer near computer', (10, 900), cv2.QT_FONT_NORMAL, 1, (102, 5, 255), 2)
            teaching_style_dict['whiteboardarea'].append(1)
            try:
                if aoi_list[4][0][0] < shoulders_mid[0] < aoi_list[4][1][0] and aoi_list[4][0][1] < shoulders_mid[1] < aoi_list[4][1][1]:
                    cv2.putText(output_image, 'Lecturer near computer', (10, 900), cv2.QT_FONT_NORMAL, 1, (102, 5, 255), 2)
                    teaching_style_dict['whiteboardarea'].append(1)
            except:
                pass
        else:
            teaching_style_dict['whiteboardarea'].append(0)


    else:
        print('Feet not found')
        teaching_style_dict['slidesarea'].append(0)
        teaching_style_dict['studentsarea'].append(0)





    return output_image, center_of_gravity



## For video
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.2, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
# video = cv2.VideoCapture(0)

# Initialize the VideoCapture object to read from a video stored in the disk.
video = cv2.VideoCapture(video_name)

# Find out FPS
video_fps = video.get(cv2.CAP_PROP_FPS)
print('Video Frame Rate: ', video_fps)

# Get total number of frames
total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frame_count)
running_frame_count = 0
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Define a codec and create a Videowriter
# Get frame width and height
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mov',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))
# print('initial', frame_width,frame_height)


## Create a list to hold the center of gravity coordinates (track teacher movement within classroom)
list_of_COG = []

# Create a holder for previous landmarks (check back, e.g., 60 frames)
landmarks_history = []

# Iterate until the video is accessed successfully.
while video.isOpened():
    start_time = datetime.datetime.now()
    # Read a frame.
    ok, frame = video.read()

    if ok == False:
        break

    # Get the width and height of the frame
    # frame_height, frame_width, _ = frame.shape


    # Print current frames / total frames
    running_frame_count += 1

    print(f"{running_frame_count}/{total_frame_count} total frames analyzed.")

    # Check if frame is not read properly.
    if not ok:
        # Break the loop.
        break

    # Flip the frame horizontally for natural (selfie-view) visualization.
    # frame = cv2.flip(frame, 1)

    # # Resize the frame while keeping the aspect ratio.
    # frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # Perform Pose landmark detection.
    frame, landmarks, landmark_details = detectPose(frame, pose_video)

    # Check if the landmarks are detected.
    if len(landmarks) > 0:
        landmarks_history.append(landmarks)

        if len(landmarks_history) > 60:
            # remove first
            landmarks_history.pop(0)


        # Perform the Pose Classification.
        frame, _, teaching_style_dict = classifyPose(landmarks, frame, landmarks_history, running_frame_count)

        ## Calculate shoulder angle and display it on video
        # right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        #                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        #                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        # cv2.putText(frame, str(right_shoulder_angle), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0) , 2)
        # print(right_shoulder_angle)

        frame, _ = trackCOG(landmarks, frame, areasofinterest_list)


    # Put in areas of interest
    aoi_bool = True
    aoilabels_list = ['Teaching Slides', 'Students', 'Computer', 'Whiteboard']

    label_index = 0
    for lm in areasofinterest_list:
        if label_index > 3:
            label = 'Whiteboard'
        else:
            label = aoilabels_list[label_index]

        overlay = frame.copy()
        alpha = 0.85
        cv2.rectangle(frame, lm[0], lm[1], (0, 0, 255), -1)
        cv2.putText(frame, f"{label}", (int((lm[1][0] + lm[0][0]) / 2), int((lm[1][1] + lm[0][1]) / 2)),
                    cv2.QT_FONT_NORMAL, 0.8, (255, 255, 255), 1)

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        label_index+=1

    """
    This part has been modified!
    14 Feb 2024
    """

    # Blur certain part of video ### 29 Jan addition
    ## Define how much of video height you want to blur (as a percentage, from the bottom up)
    blur_height_perc = 25

    blur_height_value = int(frame_height - ((blur_height_perc / 100) * frame_height))

    # Create ROI coordinates
    topLeft = (0, blur_height_value)
    bottomRight = (frame_width, frame_height)
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

    # Grab ROI with Numpy slicing and blur
    ROI = frame[y:y + h, x:x + w]
    blur = cv2.GaussianBlur(ROI, (55, 55), 0)
    frame[y:y + h, x:x + w] = blur

    ### Export the landmarks and confidence into another CSV
    print('landmarks here:')
    print(landmarks)

    if len(landmarks) > 0:
        count = 0
        for lm in landmark_details.pose_landmarks.landmark:
            # Iterate over the detected landmarks.
            # Append the landmark into the list.
            landmark_coordinate_dict[f"lm{count}_x"].append(int(lm.x * frame_width))
            landmark_coordinate_dict[f"lm{count}_y"].append(int(lm.y * frame_height))
            landmark_coordinate_dict[f"lm{count}_z"].append(int(lm.z * frame_width))
            landmark_coordinate_dict[f"lm{count}_visibility"].append(lm.visibility)

            count+=1

    temp_coord_df = pd.DataFrame.from_dict(landmark_coordinate_dict)
    temp_coord_df.to_csv("output/reportsourcefile_landmarkcoordinates.csv", index=False)

    ### Landmark integer to body part conversion here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

    """
    Until here
    14 Feb 2024
    """

    # Display the frame.
    cv2.imshow('Pose Detection', frame)

    # Calculate time taken for analyzing/showing frame
    end_time = datetime.datetime.now()
    delta = end_time - start_time

    frames_left_to_analyze = total_frame_count - running_frame_count
    print('Time taken to analyze frame:', delta.total_seconds())
    print('Estimated time left:', f"{round((frames_left_to_analyze * delta.total_seconds() / 60), 1)} mins // {frames_left_to_analyze} frames left to analyze")



    # Write the frame to video if you want
    out.write(frame)


    ## Check that the analysis of each frame is updated correctly in the teachingstyles dataframe
    arrayslist = []
    for key, value in teaching_style_dict.items():
        arrayslist.append(len(value))
    listofcounts = set(arrayslist)

    if len(listofcounts) > 1:
        arraysdict = {}
        for x in listofcounts:
            arraysdict[x] = arrayslist.count(x)

        sortedlist = sorted(arraysdict)
        print('ERROR HERE')
        print(sortedlist)

        for key, value in teaching_style_dict.items():
            if len(value) == sortedlist[0]:
                value.append(0)







    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(list_of_COG)
        break


        # k = cv2.waitKey(1) & amp; 0xFF
        #
        # # Check if 'ESC' is pressed.
        # if (k == 27):
        #     # Break the loop.
        #     break

# Release the VideoCapture object.
video.release()

# Release Video output object
# out.release()

# Close the windows.
cv2.destroyAllWindows()


# Save the list of COG coordinates
list_of_COG_x = [x[0] for x in list_of_COG]
list_of_COG_y = [x[1] for x in list_of_COG]
df = pd.DataFrame({'x':list_of_COG_x, 'y':list_of_COG_y})
df.to_csv("output/reportsourcefile_center_of_gravity.csv", index=False)


# Save the dictionary of teaching style x frame number to a dictionary -> and then to CSV
for key, value in teaching_style_dict.items():
    print(key, len(value))
teaching_style_df = pd.DataFrame.from_dict(teaching_style_dict)
teaching_style_df.to_csv("output/teachingstyle_output.csv", index=False)