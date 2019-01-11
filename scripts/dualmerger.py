# -*- coding: utf-8 -*-
import cv2
import numpy as np
import re, ast
import csv
from gaze_csv_processor import gaze_csv_processor, util, point
from audioprocessor import audioprocessor
import subprocess
import sys
import os

m_to_screen = {}


# Requirements
# import re
# import ast

def process_string_matrix(string_matrix):
    a = re.sub('\s+', ',', string_matrix)
    a = re.sub('\[,', '[', a)
    return np.array(ast.literal_eval(a))


def process_frame(frame, m_to_screen_matrix):
    mapped_space_one = np.array(
        ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32).reshape(-1, 1, 2)
    screen_space = cv2.perspectiveTransform(mapped_space_one, m_to_screen_matrix).reshape(-1, 2)
    screen_space[:, 1] = 1 - screen_space[:, 1]
    screen_space[:, 1] *= frame.shape[0]
    screen_space[:, 0] *= frame.shape[1]
    s_0, s_1 = resolution_x, resolution_y

    # flip vertically again setting mapped_space verts accordingly
    mapped_space_scaled = np.array(((0, s_1), (s_0, s_1), (s_0, 0), (0, 0)), dtype=np.float32)
    M = cv2.getPerspectiveTransform(screen_space, mapped_space_scaled)

    # perspective transformation
    srf_in_video = cv2.warpPerspective(frame, M, (int(resolution_x), int(resolution_y)))
    return srf_in_video


try:
    with open(util.get_fullpath_by_prefix('./000/surfaces/', 'srf_positons'), 'r') as f:
        data = list(csv.reader(f, delimiter=','))
except OSError as e:
    print("One of the files not found, details ", e.filename)
    sys.exit(1)
m_to_screen_string = np.array(data).T[2]
for line in np.array(data)[1:]:
    m_to_screen[int(line[0])] = process_string_matrix(line[2])
# for i in range(1,len(m_to_screen_string)-1):
# m_to_screen.append(process_string_matrix(m_to_screen_string[i]))

# Gaze data processing
base_gaze, second_gaze = gaze_csv_processor.process_gaze_data()
frame_to_timestamp = np.load("000/world_viz_timestamps.npy")
duration = frame_to_timestamp[-1] - frame_to_timestamp[0]
print(frame_to_timestamp)
try:
    with open('configs.csv', 'r') as f:
        configs_row = list(csv.reader(f, delimiter=','))
except OSError as e:
    print("Configuration file not found, ", e.filename)
    sys.exit(1)
config = np.array(configs_row)

# Config
try:
    base_radius = int(config[1][0])
    base_width = int(config[1][1])
    base_color = (int(config[1][2]), int(config[1][3]), int(config[1][4]))
    base_inner = int(config[1][5])

    second_radius = int(config[1][6])
    second_width = int(config[1][7])
    second_color = (int(config[1][8]), int(config[1][9]), int(config[1][10]))
    second_inner = int(config[1][11])

    gazes_limit = int(config[1][12])

    base_gaze_adjustment = (float(config[1][13]), float(config[1][14]))
    second_gaze_adjustment = (float(config[1][15]), float(config[1][16]))

    with_audio = int(config[1][17]) == 1
    audiogramm_length = float(config[1][18])
    need_set_of_frames = int(config[1][19]) == 1
    decomposition_quality = int(config[1][20])
    resolution_x = int(config[1][21])
    resolution_y = int(config[1][22])

except:
    print("config.csv is corrupted")
    sys.exit(1)

for plist in base_gaze.values():
    for p in plist:
        p.x += base_gaze_adjustment[0]
        p.y += base_gaze_adjustment[1]

for plist in second_gaze.values():
    for p in plist:
        p.x += second_gaze_adjustment[0]
        p.y += second_gaze_adjustment[1]

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_path_input = "./000/world_viz.mp4"
cap = cv2.VideoCapture(video_path_input)

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_path_output = "video_result.avi"
# out = cv2.VideoWriter(video_path_output, fourcc, video_fps, (1366,868))
out = cv2.VideoWriter(video_path_output, fourcc, video_fps, (resolution_x, resolution_y + 100))

if not cap.isOpened():
    print("Error opening video stream or file")


# Read until video is completed


def reduce_spectre(spectre, n):
    reduced_spectre = []
    for i in range(0, int(len(full_spectre) / n)):
        sum = 0
        for j in range(0, n):
            sum += spectre[i * n + j]
        reduced_spectre.append(int(sum / n))
    return reduced_spectre


def get_current_spectre(EoI, spectre, AFdelta):
    start_interval = max(int(EoI - AFdelta), 0)
    end_interval = int(EoI)
    spectre_raw = []
    for i in range(start_interval, end_interval):
        spectre_raw.append(np.array([int(1.0 * resolution_x * (i - start_interval) / AFdelta),
                                     int(spectre[i] / spectre_max * 50) + resolution_y + (100 / 2)]))
    spectre_formatted = np.array(spectre_raw, np.int32)
    spectre_formatted = spectre_formatted.reshape((-1, 1, 2))
    return spectre_formatted


audio_path_output = "audio_from_world_viz.wav"
if with_audio:
    # Get audio from  world_viz
    subprocess.call(['ffmpeg', '-y', '-i', video_path_input, '-codec:a', 'pcm_s16le', '-ac', '1', audio_path_output])
    # cmd = 'ffmpeg -y -i ' + video_path_input + ' -codec:a pcm_s16le -ac 1' + audio_path_output
    # subprocess.call(cmd)
    ap = audioprocessor(audio_path_output)
    full_spectre = ap.get_spectrum()

    print("lenght of full_spectre ", len(full_spectre))
    full_spectre = reduce_spectre(full_spectre, 100)
    print("lenght of reduced full_spectre ", len(full_spectre))
    FPS = video_fps
    AFPF = len(full_spectre) / len(base_gaze)
    delta = audiogramm_length
    AFdelta = delta * FPS * AFPF

    spectre_max = np.max(full_spectre)
    # spectre_raw = []
    left = 0
    right = len(full_spectre)


def get_nearest_point(points, original_point: point) -> point:
    last_delta = 9999999
    current_answer = None
    # for point in points[original_point.vf]:
    #     delta = abs(original_point.ts-point.ts)
    #     if delta < last_delta:
    #         last_delta = delta
    #         current_answer = point
    #     elif delta > last_delta:
    #         break
    # return current_answer
    for point_list in points.values():
        for point in point_list:
            delta = abs(original_point.ts - point.ts)
            if delta < last_delta:
                last_delta = delta
                current_answer = point
            elif delta > last_delta:
                break
    return current_answer


def create_folder(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_current_spectre_2(spectre, AFdelta, current_frame):
    end_time = frame_to_timestamp[current_frame] - frame_to_timestamp[0]
    start_time = end_time - audiogramm_length

    end_interval = int(len(full_spectre) * end_time / duration)
    start_interval = max(int(len(full_spectre) * start_time / duration), 0)

    AFdelta = audiogramm_length * FPS * len(full_spectre) / len(base_gaze)

    spectre_raw = []
    for i in range(start_interval, end_interval):
        spectre_raw.append(np.array([int(1.0 * resolution_x * (i - start_interval) / AFdelta),
                                     int(spectre[i] / spectre_max * 50) + resolution_y + (100 / 2)]))
    spectre_formatted = np.array(spectre_raw, np.int32)
    spectre_formatted = spectre_formatted.reshape((-1, 1, 2))
    return spectre_formatted


gazes_hist_list_base = []
gazes_hist_list_second = []

with open('frames_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'video_frame', 'first_x', 'first_y', 'second_x', 'second_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for k, base_gazes_frame in base_gaze.items():
        # print(base_gazes_frame)
        for current_base_gaze in base_gazes_frame:
            nearest_second_gaze = get_nearest_point(second_gaze, current_base_gaze)
            writer.writerow({'timestamp': current_base_gaze.ts,
                             'video_frame': current_base_gaze.vf,
                             'first_x': current_base_gaze.x,
                             'first_y': current_base_gaze.y,
                             'second_x': nearest_second_gaze.x,
                             'second_y': nearest_second_gaze.y})

m_to_screen_standart = np.array([[0.39129562, -0.01764555, 0.32704431],
                                 [0.03696206, 0.51439033, 0.2158329],
                                 [0.010974, 0.0118955, 1.]])
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(base_gaze)
for i in range(len(frame_to_timestamp)):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Processing notifications
        if i % 50 == 0:
            print("Frame ", i, " out of ", len(frame_to_timestamp))
        if i % 1 == 0:
            frameNew = process_frame(frame, m_to_screen_standart)
            if i in base_gaze:
                frameNew = process_frame(frame, m_to_screen[i])
                gaze: point
                for gaze in base_gaze[i]:
                    if 0 <= gaze.x and gaze.x <= 1 and 0 <= gaze.y and gaze.y <= 1:
                        gazes_hist_list_base.append(gaze)
                        if len(gazes_hist_list_base) > gazes_limit:
                            gazes_hist_list_base.pop(0)
                if len(base_gaze[i]) == 0 and len(gazes_hist_list_base) > 0:
                    gazes_hist_list_base.pop(0)

                for gaze in second_gaze[i]:
                    if gaze.x >= 0 and gaze.x <= 1 and gaze.y >= 0 and gaze.y <= 1:
                        gazes_hist_list_second.append(gaze)
                        if len(gazes_hist_list_second) > gazes_limit:
                            gazes_hist_list_second.pop(0)
                if len(second_gaze[i]) == 0 and len(gazes_hist_list_second) > 0:
                    gazes_hist_list_second.pop(0)
                # Base gazes drawing
                last_gaze = None
                for current_gaze in reversed(gazes_hist_list_base):
                    if last_gaze != None:
                        cv2.line(frameNew, (int(resolution_x * last_gaze.x), int(resolution_y * (1 - last_gaze.y))),
                                 (int(resolution_x * current_gaze.x), int(resolution_y * (1 - current_gaze.y))),
                                 base_color, base_width)
                    elif len(base_gaze[i]) != 0:
                        cv2.circle(frameNew,
                                   (int(resolution_x * current_gaze.x), int(resolution_y * (1 - current_gaze.y))),
                                   int(base_radius * 2), base_color, 3)
                    cv2.circle(frameNew, (int(resolution_x * current_gaze.x), int(resolution_y * (1 - current_gaze.y))),
                               int(base_radius), base_color, -1)
                    last_gaze = current_gaze

                # Second gazes drawing
                last_gaze = None
                for current_gaze in reversed(gazes_hist_list_second):
                    if last_gaze != None:
                        cv2.line(frameNew, (int(resolution_x * last_gaze.x), int(resolution_y * (1 - last_gaze.y))),
                                 (int(resolution_x * current_gaze.x), int(resolution_y * (1 - current_gaze.y))),
                                 second_color, second_width)
                    elif len(second_gaze[i]) != 0:
                        cv2.circle(frameNew,
                                   (int(resolution_x * current_gaze.x), int(resolution_y * (1 - current_gaze.y))),
                                   int(second_radius * 2), second_color, 3)
                    cv2.circle(frameNew, (int(resolution_x * current_gaze.x), int(resolution_y * (1 - current_gaze.y))),
                               int(second_radius), second_color, -1)
                    last_gaze = current_gaze
            else:
                cv2.rectangle(frame, (0, 0), (resolution_x, resolution_y), (0, 255, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frameNew, i.__str__(), (int(resolution_x * 0.87), int(resolution_y * 0.87)), font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Draw audiogramm (audio spectre)
            frameWithFooter = cv2.copyMakeBorder(frameNew, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if with_audio:
                # End of Interval (on the screen)
                # EoI = len(full_spectre) * i / len(base_gaze)
                # cv2.polylines(frameWithFooter, [get_current_spectre(EoI, full_spectre, AFdelta)], False, (0, 0, 255))

                cv2.polylines(frameWithFooter, [get_current_spectre_2(full_spectre, AFdelta, i)], False, (0, 0, 255))
            out.write(frameWithFooter)
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
if with_audio:
    # cmd = 'ffmpeg -y -i '+video_path_output+' -i '+ audio_path_output + ' -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 '+' dual_'+video_path_output

    # cmd1 = 'ffmpeg -y -i ' + video_path_output + ' -i '+ audio_path_output + ' -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 ' + ' 1fin_'+video_path_output
    # subprocess.call(cmd1)
    subprocess.call(
        ['ffmpeg', '-y', '-i', video_path_input, '-i', audio_path_output, '-c:v', 'copy', '-c:a', 'acc', '-map',
         '0:v:0', '-map', '1:a:0', 'high_quality_with_audio_' + video_path_output])

    # cmd2 ="ffmpeg -y -i " + audio_path_output + " -i " + video_path_output + " low_quality_with_audio_" + video_path_output
    # subprocess.call(cmd2)
    subprocess.call(['ffmpeg', '-y', '-i', audio_path_output, '-i', video_path_input,
                     'low_quality_with_audio_' + video_path_output])
    # cmd = 'ffmpeg -y -i ' + video_path_output + ' ./frames%04d.jpg'
    # subprocess.call(cmd)

if need_set_of_frames:
    create_folder('frames')
    subprocess.call(
        ['ffmpeg', '-i', 'video_result.avi', '-q:v', str(decomposition_quality), os.path.join('frames', 'image%d.jpg')])
    # cmd_to_extract_frames = 'ffmpeg -i video_result.avi -q:v '+str(decomposition_quality)+' frames/image%d.jpg'
    # subprocess.call(cmd_to_extract_frames)
    print('Set of frames has been extracted')
os.remove(video_path_output)
print("Done!")
# os.remove(audio_path_output)
# os.remove(video_path_output)
# return np.array(ast.literal_eval(a))
