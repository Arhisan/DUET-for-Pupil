# -*- coding: utf-8 -*-
from bisect import bisect_left
from typing import List, Set, Dict, Tuple, Optional
import cv2
import numpy as np
import re, ast
import csv
from gaze_csv_processor import gaze_csv_processor, util, Point
from audioprocessor import AudioProcessor
import subprocess
import sys
import os
import json

from library import get_nearest_point

m_to_screen = {}
exposed_dir = "data"

# Requirements
# import re
# import ast

def process_string_matrix(string_matrix: str) -> np.array:
    a = re.sub('\s+', ',', string_matrix)
    a = re.sub('\[,', '[', a)
    try:
        return np.array(ast.literal_eval(a))
    except ValueError:
        print(f"Error value in matrix {a}, nan's replaced with 1.")
        a = re.sub('nan', '1.', a)
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

# Config

try:
    with open(os.path.join(exposed_dir, 'config.json')) as config_file:
        cfg = json.load(config_file)

        marker = cfg.get("marker")

        base_color = marker.get("base_color")
        base_gaze_adjustment = marker.get("base_gaze_adjustment")
        second_color = marker.get("second_color")
        second_gaze_adjustment = marker.get("second_gaze_adjustment")
        output_resolution = cfg.get("output_resolution")

        base_radius:                int = int(marker.get("base_radius"))
        base_width:                 int = int(marker.get("base_line_width"))
        base_color:                 (int, int, int) = (int(base_color.get("r")), int(base_color.get("g")), int(base_color.get("b")))
        base_inner:                 int = int(marker.get("base_inner_radius"))
        base_gaze_adjustment:       (float, float) = (float(base_gaze_adjustment.get("x")), float(base_gaze_adjustment.get("y")))

        second_radius:              int = int(marker.get("second_radius"))
        second_width:               int = int(marker.get("second_line_width"))
        second_color:               (int, int, int) = (int(second_color.get("r")), int(second_color.get("g")), int(second_color.get("b")))
        second_inner:               int = int(marker.get("second_inner_radius"))
        second_gaze_adjustment:     (float, float) = (float(second_gaze_adjustment.get("x")), float(second_gaze_adjustment.get("y")))

        gazes_limit:                int = int(marker.get("gazes_limit"))

        with_audio:                 bool = int(cfg.get("with_audio")) == 1
        audiogramm_length:          float = float(cfg.get("waveform_length"))
        need_set_of_frames:         bool = int(cfg.get("decomposition_to_set_of_frames")) == 1
        decomposition_quality:      int = int(cfg.get("frames_quality"))
        resolution_x:               int = int(output_resolution.get("x"))
        resolution_y:               int = int(output_resolution.get("y"))

        paths = cfg.get("paths")
        base_dir:                   str = paths.get("base_dir", "000")
        second_dir:                 str = paths.get("second_dir", "001")
        video_path_input:           str = paths.get("video_path", "000/world.mp4")
        outer_audio_path:           str = paths.get("outer_audio_path")
        outer_audio_timestamp_path: str = paths.get("outer_audio_timestamp_path")

except OSError as e:
    print("Configuration file not found, ", e.filename)
    sys.exit(1)

try:
    path = os.path.join(exposed_dir, base_dir, 'surfaces/')
    print (path)
    with open(util.get_fullpath_by_prefix(path, 'srf_positons'), 'r') as f:
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
base_gaze, second_gaze = gaze_csv_processor.process_gaze_data(path0=os.path.join(exposed_dir,base_dir,'surfaces/'), path1 = os.path.join(exposed_dir,second_dir,'surfaces/'))
base_gaze_flatten = np.array(base_gaze.values()).flatten()
second_gaze_flatten = np.array(second_gaze.values()).flatten()

frame_to_timestamp = np.load(os.path.join(exposed_dir, base_dir, 'world_timestamps.npy'))
duration = frame_to_timestamp[-1] - frame_to_timestamp[0]
print(frame_to_timestamp)

for plist in base_gaze.values():
    for p in plist:
        p.x += base_gaze_adjustment[0]
        p.y += base_gaze_adjustment[1]

for plist in second_gaze.values():
    for p in plist:
        p.x += second_gaze_adjustment[0]
        p.y += second_gaze_adjustment[1]

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap = cv2.VideoCapture(os.path.join(exposed_dir, video_path_input))

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_path_output_base = "video_result.avi"
video_path_output = os.path.join(exposed_dir, video_path_output_base)
# out = cv2.VideoWriter(video_path_output, fourcc, video_fps, (1366,868))
out = cv2.VideoWriter(video_path_output, fourcc, video_fps, (resolution_x, resolution_y + 100))

if not cap.isOpened():
    print("Error opening video stream or file")


# Read until video is completed

def reduce_spectre(spectre: List[int], n: int) -> List[int]:
    reduced_spectre = []
    for i in range(0, int(len(full_spectre) / n)):
        sum = 0
        for j in range(0, n):
            sum += spectre[i * n + j]
        reduced_spectre.append(int(sum / n))
    return reduced_spectre


def get_current_spectre(EoI, spectre, AFdelta) -> np.array:
    start_interval = max(int(EoI - AFdelta), 0)
    end_interval = int(EoI)
    spectre_raw = []
    for i in range(start_interval, end_interval):
        spectre_raw.append(np.array([int(1.0 * resolution_x * (i - start_interval) / AFdelta),
                                     int(spectre[i] / spectre_max * 50) + resolution_y + (100 / 2)]))
    spectre_formatted = np.array(spectre_raw, np.int32)
    spectre_formatted = spectre_formatted.reshape((-1, 1, 2))
    return spectre_formatted

audio_path_output_base = "audio_from_world_viz.wav"
audio_path_output = os.path.join(exposed_dir, audio_path_output_base)

if with_audio:
    # Get audio from  world_viz
    if outer_audio_path is not None:
        print("Using outer audio source:%s\n", outer_audio_path)
        subprocess.call(['ffmpeg', '-y', '-i', os.path.join(exposed_dir, outer_audio_path), '-codec:a', 'pcm_s16le', '-ac', '1', audio_path_output])
    else:
        print("Using audio from video\n")
        subprocess.call(['ffmpeg', '-y', '-i', video_path_input, '-codec:a', 'pcm_s16le', '-ac', '1', audio_path_output])
    # cmd = 'ffmpeg -y -i ' + video_path_input + ' -codec:a pcm_s16le -ac 1' + audio_path_output
    # subprocess.call(cmd)
    ap = AudioProcessor(audio_path_output)
    full_spectre = ap.get_spectrum()

    print("lenght of full_spectre ", len(full_spectre))
    full_spectre = reduce_spectre(full_spectre, 100)
    print("lenght of reduced full_spectre ", len(full_spectre))
    FPS = video_fps
    AFPF = len(full_spectre) / len(frame_to_timestamp)
    delta = audiogramm_length
    AFdelta = delta * FPS * AFPF

    spectre_max = np.max(full_spectre)
    # spectre_raw = []
    left = 0
    right = len(full_spectre)


def create_folder(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_current_spectre_2(spectre: List[int], current_frame: int) -> np.array:
    end_time:   float = frame_to_timestamp[current_frame] - frame_to_timestamp[0]
    start_time: float = end_time - audiogramm_length

    end_interval:   int = int(len(full_spectre) * end_time / duration)
    start_interval: int = max(int(len(full_spectre) * start_time / duration), 0)

    AFdelta: float = audiogramm_length * FPS * len(full_spectre) / len(frame_to_timestamp)

    spectre_raw: List[np.array] = []
    for i in range(start_interval, end_interval):
        spectre_raw.append(np.array([int(1.0 * resolution_x * (i - start_interval) / AFdelta),
                                     int(spectre[i] / spectre_max * 50) + resolution_y + (100 / 2)]))
    spectre_formatted: np.array = np.array(spectre_raw, np.int32).reshape((-1, 1, 2))
   # spectre_formatted = spectre_formatted.reshape((-1, 1, 2))
    return spectre_formatted


gazes_hist_list_base:   List[Point] = []
gazes_hist_list_second: List[Point] = []

with open(os.path.join(exposed_dir, 'frames_data.csv'), 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'video_frame', 'first_x', 'first_y', 'second_x', 'second_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # let's create helping flattened second_gaze
    second_gaze_flattened = []
    for points in second_gaze.values():
        for point in points:
            second_gaze_flattened.append(point)
    second_gaze_flattened.sort(key=lambda x: x.ts)
    # end of flattened second_gaze creation

    for k, base_gazes_frame in base_gaze.items():
        # print(base_gazes_frame)
        for current_base_gaze in base_gazes_frame:
            nearest_second_gaze = get_nearest_point(second_gaze_flattened, current_base_gaze)
            writer.writerow({'timestamp': current_base_gaze.ts,
                             'video_frame': current_base_gaze.vf,
                             'first_x': current_base_gaze.x,
                             'first_y': current_base_gaze.y,
                             'second_x': nearest_second_gaze.x,
                             'second_y': nearest_second_gaze.y})

m_to_screen_standard = np.array([[0.39129562, -0.01764555, 0.32704431],
                                 [0.03696206,   0.51439033, 0.2158329],
                                 [0.010974,     0.0118955,  1.0000000]])

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for i in range(len(frame_to_timestamp)):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Processing notifications
        if i % 20 == 0:
            print("Frame ", i, " out of ", len(frame_to_timestamp))
        if i % 1 == 0:
            frameNew = None
            if i in base_gaze:
                frameNew = process_frame(frame, m_to_screen[i])
                gaze: Point
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
                #cv2.rectangle(frame, (0, 0), (resolution_x, resolution_y), (0, 255, 0), -1)
                frameNew = process_frame(frame, m_to_screen_standard)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frameNew, i.__str__(), (int(resolution_x * 0.87), int(resolution_y * 0.87)), font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Draw audiogramm (audio spectre)
            frameWithFooter = cv2.copyMakeBorder(frameNew, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if with_audio:
                # End of Interval (on the screen)
                # EoI = len(full_spectre) * i / len(base_gaze)
                # cv2.polylines(frameWithFooter, [get_current_spectre(EoI, full_spectre, AFdelta)], False, (0, 0, 255))

                cv2.polylines(frameWithFooter, [get_current_spectre_2(full_spectre, i)], False, (0, 0, 255))
            out.write(frameWithFooter)
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()

if with_audio:
    # cmd = 'ffmpeg -y -i '+video_path_output+' -i '+ audio_path_output + ' -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 '+' dual_'+video_path_output
    #cmd1 = 'ffmpeg -y -i ' + video_path_output + ' -i '+ audio_path_output + ' -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 ' + ' ' + os.path.join(exposed_dir, '1fin_' + video_path_output_base)
    #subprocess.call(cmd1)

    subprocess.call(
        ['ffmpeg', '-y', '-i', video_path_output, '-i', audio_path_output, '-c:v', 'copy', '-c:a', 'acc', '-map',
         '0:v:0', '-map', '1:a:0', os.path.join(exposed_dir, '1fin_' + video_path_output_base)])

    #subprocess.call(
    #    ['ffmpeg', '-y', '-i', video_path_input, '-i', audio_path_output, '-c:v', 'copy', '-c:a', 'acc', '-map',
    #     '0:v:0', '-map', '1:a:0', 'high_quality_with_audio_' + video_path_output])

    # cmd2 ="ffmpeg -y -i " + audio_path_output + " -i " + video_path_output + " low_quality_with_audio_" + video_path_output
    # subprocess.call(cmd2)
    subprocess.call(['ffmpeg', '-y', '-i', audio_path_output, '-i', video_path_output,
                     os.path.join(exposed_dir, 'low_quality_with_audio_' + video_path_output_base)])
    # cmd = 'ffmpeg -y -i ' + video_path_output + ' ./frames%04d.jpg'
    # subprocess.call(cmd)

if need_set_of_frames:
    create_folder(os.path.join(exposed_dir,'frames'))
    subprocess.call(
        ['ffmpeg', '-i', os.path.join(exposed_dir,'video_result.avi'), '-q:v', str(decomposition_quality), os.path.join(exposed_dir, 'frames', 'image%d.jpg')])
    # cmd_to_extract_frames = 'ffmpeg -i video_result.avi -q:v '+str(decomposition_quality)+' frames/image%d.jpg'
    # subprocess.call(cmd_to_extract_frames)
    print('Set of frames has been extracted')
#os.remove(video_path_output)
print("Done!")
# os.remove(audio_path_output)
# os.remove(video_path_output)
# return np.array(ast.literal_eval(a))
