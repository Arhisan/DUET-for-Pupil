# DUET for Pupil


**DUET  (DUal Eye-Tracking) for Pupil** - is an independent utility for processing paired dual eye-tracking [Pupil-labs](https://github.com/pupil-labs/pupil) projects. 

### Requirements:
- Python3
- NumPy
- cv2 (openCV python wrapper)
- FFMpeg

### Content:
- dualmerger.py (executable)
- audioprocessor.py
- gaze_csv_processor.py
- config.csv

## Get Started
Prepare Pupil-labs projects created while dual eye-tracking session.

1. Open both projects via Pupil-player and export the desired surface from each.
2. Extract export folders into one directory (i.e. "dir/") and rename them with "000" and "001" as follows.
3. Put all the files of the utility from this repo to "dir/"
4. Set the config.csv 
5. Run dualmerger.py with python


### Config Structure
| Fields                     |  Description  |
|------------------------------------|-------------|
| base_radius, base_line_width, base_inner_radius, second_radius, second_line_width, second_inner_radius | Gaze-track width prorepties. The newest gaze point draws with circle with "xxx_radius". Use "xxx_inner_radius" to make the ring instead of circle. |
| base_color_red, base_color_green, base_color_blue, second_color_red, second_color_green, second_color_blue | Gaze-track color prorepties. Set all the components with values from 0 to 255. |
| gazes_limit                                                                                         | Number of the gazes in the tail of the track. |
| base_gaze_adjustment_x, base_gaze_adjustment_y, second_gaze_adjustment_x, second_gaze_adjustment_y    | Adjustment. |
| with_audio, waveform_length                                                                             | Flag - if audioprocessing is needed (=1) or not (=0). "waveform_length" is the length of the drawing audiospectre in seconds. |
| decomposition_to_set_of_frames, frames_quality                                                           | Flag - if decomposition to the set of frames is needed (=1) or not (=0). "frames_quality" is the quality pf the decomposed frames (from 1 to 31).  |
| resolution_x, resolution_y                        |  The resolution of the output video.  |
