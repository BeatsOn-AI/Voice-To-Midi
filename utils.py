import librosa
import vamp
import argparse
import os
import numpy as np
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
#import jams
import __init__
import pretty_midi
import matplotlib.pyplot as plt


def save_jams(jamsfile, notes, track_duration, orig_filename):

    # Construct a new JAMS object and annotation records
    jam = jams.JAMS()

    # Store the track duration
    jam.file_metadata.duration = track_duration
    jam.file_metadata.title = orig_filename

    midi_an = jams.Annotation(namespace='pitch_midi',
                              duration=track_duration)
    midi_an.annotation_metadata = \
        jams.AnnotationMetadata(
            data_source='audio_to_midi_melodia.py v%s' % __init__.__version__,
            annotation_tools='audio_to_midi_melodia.py (https://github.com/'
                             'justinsalamon/audio_to_midi_melodia)')

    # Add midi notes to the annotation record.
    for n in notes:
        midi_an.append(time=n[0], duration=n[1], value=n[2], confidence=0)

    # Store the new annotation in the jam
    jam.annotations.append(midi_an)

    # Save to disk
    jam.save(jamsfile)

def save_midi(outfile, notes, tempo):

    track = 0
    time = 0
    midifile = MIDIFile(1)

    # Add track name and tempo.
    midifile.addTrackName(track, time, "MIDI TRACK")
    midifile.addTempo(track, time, tempo)

    channel = 0
    volume = 100

    for note in notes:
        onset = note[0] * (tempo/60.)
        duration = note[1] * (tempo/60.)
        # duration = 1
        pitch = note[2]
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()

def hz2midi(hz):
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1  # Ensure non-negative values
    midi = 69 + 12 * np.log2(hz_nonneg / 440.0)
    midi[idx] = -1  # Assign an invalid placeholder for non-audible frequencies

    # Validate and round MIDI note numbers
    midi = np.where((midi >= 0) & (midi <= 127), np.round(midi), -1).astype(int)
    
    return midi

def convert_to_onset_duration_pprev(notes, hop_length, fs):
    converted_notes = []
    
    for onset, offset, pitch in notes:
        onset_sec = onset * hop_length / float(fs)
        duration_sec = (offset - onset + 1) * hop_length / float(fs)
        converted_notes.append((onset_sec, duration_sec, pitch))
    
    return converted_notes


def convert_to_onset_offset_pitch(notes, hop_length, fs):
    converted_notes = []
    
    for i, (onset_sec, duration_sec, pitch) in enumerate(notes):
        onset = int(onset_sec * fs / hop_length)
        offset = onset + int(duration_sec * fs / hop_length) - 1
        converted_notes.append((onset, offset, pitch))
    
    return converted_notes



def midi_to_piano_roll_image(midi_file_path, output_image_path):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Get the piano roll
    piano_roll = midi_data.get_piano_roll()

    # Plot the piano roll
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(piano_roll, aspect='auto', origin='lower', cmap='gray_r')
    ax.set_xlabel('Time (in MIDI ticks)')
    ax.set_ylabel('MIDI Note Number')
    ax.set_title('Piano Roll')

    # Save the figure
    plt.savefig(output_image_path)
    plt.close(fig)

def freq2cent(freq):
    # Base pitch reference is A4 at 440 Hz
    cent = []
    for f in freq:
        if f > 0:
            # Calculate cent value for valid positive frequencies
            cent_value = 1200*np.log2(f/(440. * np.power(2., 3./12.-5)))
            rounded_cent_value = round(cent_value,2)
            cent.append(rounded_cent_value)
        else:
            # Handle non-positive frequencies by appending a specific invalid value (None or a special code)
            cent.append(np.nan)  # You can use 'None' or a specific error value like -1 or np.nan depending on your needs
    return cent

def cent2freq(cent):
    freq = []
    for i in range(len(cent)):
        freq.append((440. * np.power(2., 3./12.-5)) * np.power(2., cent[i]/1200.))
    return freq

def histogram_mean(onset_candidate, offset_candidate, full_pitch):
    # Exclude the first and last 5% to remove boundary outliers
    total_length = offset_candidate - onset_candidate + 1
    valid_start = onset_candidate + int(total_length * 0.05)
    valid_end = offset_candidate - int(total_length * 0.05) + 1
    #print(valid_start,valid_end)
    full_pitch = full_pitch[~np.isnan(full_pitch)]
    current_pitch = np.array(full_pitch[valid_start:valid_end])

    min_pitch = np.min(current_pitch)
    max_pitch = np.max(current_pitch)
    
    if min_pitch == max_pitch:
        return min_pitch  # All pitches are the same, return the common pitch

    range_size = (max_pitch - min_pitch) / 5.0
    histogram = [0] * 5  # Five bins
    #print("range",range_size)
    for pitch in current_pitch:
        index = int((pitch - min_pitch) / range_size)
        if index == 5:  # Handle edge case where pitch equals max_pitch
            index = 4
        histogram[index] += 1

    max_index = np.argmax(histogram)
    # Calculate the mean of pitches in the most populated bin
    bin_min = min_pitch + range_size * max_index
    bin_max = min_pitch + range_size * (max_index + 1)
    pitches_in_bin = [p for p in current_pitch if bin_min <= p < bin_max]

    return np.mean(pitches_in_bin) if pitches_in_bin else np.mean(current_pitch)  # Fallback to average of current_pitch


def note_segment(time, pitch):
    # time is the X-axis, pitch is the Y-axis
    threshold = 170.
    window_size = 15
    onset_candidate = []
    offset_candidate = []

    num = 0
    while num < len(time) - window_size - 1:
        # num is the serial number
        cumulative_deviation = 0.
        range_max = 0.
        range_min = 0.

        for i in range(num, num + window_size):  # Add windows and calculate deviation
            if i == num:
                range_max = pitch[i]
                range_min = pitch[i]
                cumulative_deviation = 0.
            else:
                if pitch[i] > range_max:
                    cumulative_deviation = max(cumulative_deviation, pitch[i] - range_min)
                    range_max = pitch[i]
                elif pitch[i] < range_min:
                    cumulative_deviation = max(cumulative_deviation, range_max - pitch[i])
                    range_min = pitch[i]

        if cumulative_deviation > threshold or min(pitch[num: num + window_size]) < 500: # Exceeds the threshold or is somewhat in the silent zone
            num += 1
            continue
        else:
            onset_candidate.append(num)  # onset candidate is the first time value of the window
            # Scan the following points one by one
            num += window_size
            while cumulative_deviation <= threshold and num < len(time)-1:
                if pitch[num] > range_max:
                    cumulative_deviation = max(cumulative_deviation, pitch[num] - range_min)
                    range_max = pitch[num]
                elif pitch[num] < range_min:
                    cumulative_deviation = max(cumulative_deviation, range_max - pitch[num])
                    range_min = pitch[num]
                num += 1
            offset_candidate.append(num-2)  # When exceeded, this point is the offset candidate
            num -= 1

    # Note is stored in the form of triples of [onset, offset, pitch], which is a nested list
    note = []
    for i in range(len(onset_candidate)):
        note.append([onset_candidate[i], offset_candidate[i], histogram_mean(onset_candidate[i], offset_candidate[i], pitch)])
    return note

def note_postprocessing(note, full_pitch):

    # Process note merging
    # Depends on the distance between onset and offset

    cnt = 0
    length = len(note)
    while cnt < length - 1:
        delete_size = int((note[cnt][1] - note[cnt][0]) / 20)
        # The distance between the previous offset and the next onset does not exceed 5
        if note[cnt+1][0] - note[cnt][1] <= 3:
            # There are two situations that will be merged
            # 1. If the difference between pitches is less than 45, merge
            # 2. The pitch in a certain note has only one peak value (that is, the upright "V" and the inverted "V") and the difference
            # between the note pitch and the average value of all pitches is greater than 40
            # (to avoid the situation where the bottom end of V is very wide) , must be merged

            # Process the first type first
            if abs(note[cnt+1][2] - note[cnt][2]) < 45:
                note[cnt] = [note[cnt][0], note[cnt+1][1], histogram_mean(note[cnt][0], note[cnt+1][1], full_pitch)]
                del note[cnt+1]
                length -= 1

            # Process the second one again
            elif abs(note[cnt][2] - np.mean(full_pitch[note[cnt][0]:note[cnt][1]+1])) > 40:
                # Count the peak values in note
                 # Count only the previous note
                 # Scan on both sides to avoid false peak
                current_pitch = full_pitch[note[cnt][0] + delete_size: note[cnt][1]-delete_size]
                left = 0
                right = len(current_pitch)-1
                left_min = current_pitch[0]
                left_max = current_pitch[0]
                right_min = current_pitch[-1]
                right_max = current_pitch[-1]
                max_pos = 0
                min_pos = 0
                while left < right:
                    #Look at the left first
                    if current_pitch[left] < left_min and current_pitch[left] < right_min:
                        # is the current best wave trough
                        left_min = current_pitch[left]
                        min_pos = left
                    if current_pitch[left] > left_max and current_pitch[left] > right_max:
                        # is the current best wave peak
                        left_max = current_pitch[left]
                        max_pos = right
                    #Look to the right again
                    if current_pitch[right] < left_min and current_pitch[right] < right_min:
                        # is the current best wave trough
                        min_pos = right
                        right_min = current_pitch[right]
                    if current_pitch[right] > left_max and current_pitch[right] > right_max:
                        # is the current best wave peak
                        max_pos = right
                        right_max = current_pitch[right]
                    left += 1
                    right -= 1
                if min_pos * max_pos == 0:
                    if min_pos + max_pos == 0:
                        # means there are no peaks and troughs, it is monotonous
                        cnt += 1
                        continue
                    else:
                        # means there is only one peak or trough, first check whether its peak or trough is false
                         # By comparing with the pitches of the two endpoints, the difference must be greater than 50
                         # If satisfied, merge it with the one with a closer pitch in the two notes before and after it.
                         # Compare the before and after note with its pitch
                         # If it is the first or last one, merge it directly without looking at it.
                        if min(abs(current_pitch[max(min_pos, max_pos)] - current_pitch[0]), abs(current_pitch[max(min_pos, max_pos)] - current_pitch[-1])) < 50:
                            cnt += 1
                            continue
                        else:
                            if cnt - 1 <= 0 or cnt + 1 >= length - 1:
                                note[cnt] = [note[cnt][0], note[cnt+1][1], histogram_mean(note[cnt][0], note[cnt+1][1], full_pitch)]
                                del note[cnt+1]
                                length -= 1
                            else:
                                # If it is in the middle, compare pitches
                                if abs(note[cnt-1][2] - note[cnt][2]) < abs(note[cnt][2] - note[cnt+1][2]) and note[cnt][0] - note[cnt-1][1] <= 5:
                                    # Merge with the previous one. Note that what is merged at this time is cnt-1. After processing here, cnt cannot be ++.
                                    note[cnt-1] = [note[cnt-1][0], note[cnt][1], histogram_mean(note[cnt-1][0], note[cnt][1], full_pitch)]
                                    del note[cnt]
                                    length -= 1
                                    cnt -= 1
                                else:
                                    # Merge with the following
                                    note[cnt] = [note[cnt][0], note[cnt+1][1], histogram_mean(note[cnt][0], note[cnt+1][1], full_pitch)]
                                    del note[cnt+1]
                                    length -= 1
        cnt += 1

    return note

def small_note_segment(note, full_pitch, vibrato):
    # Search for notes longer than 35 points
    # I chose 35 because the length of 35 is the maximum possible length of the grace note. Such a note must be longer than the grace note.
    pos = 0
    length = len(note)
    while pos < length:
        single_note = note[pos]
        if single_note[1] - single_note[0] > 35:
            # Establish a bool array, looking for those points where the pitch differs from the note pitch by more than 65
            current_note = full_pitch[single_note[0]:single_note[1]+1]
            equal_to_note_pitch = [True] * len(current_note)
            for i in range(len(current_note)):
                if abs(current_note[i] - single_note[2]) > 65.:
                    equal_to_note_pitch[i] = False
            # Only look for all continuous regions in the points where the bool array is True
            # If there are any, see if the pitch deviation in this continuous region is within 65cent
            # The advantage of this split is that if there are large fluctuations in pitch, it will be split, so errors will be found when comparing with the standard
            # So this is a transcription method suitable for singing evaluation
            max_pitch = 0
            min_pitch = 0
            pitch_region = []
            for i in range(len(current_note)):
                # First see if it is not equal to note pitch
                if not equal_to_note_pitch[i]:
                    # Is it the first in the current region?
                    if len(pitch_region) == 0:
                        pitch_region.append(i)
                        max_pitch = current_note[i]
                        min_pitch = current_note[i]
                    else:
                        # Region is not empty, already more than one
                        # See if the range of max-min is within 65
                        if current_note[i] > max_pitch and current_note[i] - min_pitch < 65.:
                            pitch_region.append(i)
                            max_pitch = current_note[i]
                        elif current_note[i] < min_pitch and max_pitch - current_note[i] < 65.:
                            pitch_region.append(i)
                            min_pitch = current_note[i]
                        elif min_pitch < current_note[i] < max_pitch:
                            pitch_region.append(i)
                        else:
                            # The range exceeded 65, determine whether this region should be split separately
                            # First see how long this region is, if it is greater than 15 points (choose 15 because the range of decorative sounds is 15), it needs to be split, otherwise it definitely will not be split
                            if len(pitch_region) < 15:
                                pitch_region = []
                                max_pitch = 0
                                min_pitch = 0
                                i = i - len(pitch_region) + 2
                            else:
                                # Length is enough, according to the location of this region, determine whether to cut into two or three parts
                                note, pitch_region, pos, length = split_note(note, pitch_region, single_note, pos, length, full_pitch, vibrato)
                                break
                else:
                    # At this point, this point does not meet the requirements, see if the segment up to now can be cut out
                    if len(pitch_region) < 15:
                        pitch_region = []
                        max_pitch = 0
                        min_pitch = 0
                    else:
                        # Length is enough, according to the location of this region, determine whether to cut into two or three parts
                        note, pitch_region, pos, length = split_note(note, pitch_region, single_note, pos, length, full_pitch, vibrato)
                        break
            # If it is scanned to the end and the last section is long enough, cut into two parts
            if len(pitch_region) > 15:
                length += 1
                del note[pos]
                note.insert(pos, [single_note[0]+pitch_region[0], single_note[1], histogram_mean(single_note[0]+pitch_region[0], single_note[1], full_pitch)])
                note.insert(pos, [single_note[0], single_note[0]+pitch_region[0]-2, histogram_mean(single_note[0], single_note[0]+pitch_region[0]-2, full_pitch)])
                pos += 1
            else:
                # Scan the next
                pos += 1
        else:
            # Scan the next note
            pos += 1
    return note

def onset_offset_adjust(note, full_pitch):
    for cnt in range(len(note)-1):
        current_note = note[cnt]
        next_note = note[cnt+1]
        if next_note[0] - current_note[1] <= 3 and min(full_pitch[current_note[1]:next_note[0]+1]) > 0:
            # Check the area in the histogram
            # First check the current note
            current_note_pitch = full_pitch[current_note[0]:current_note[1]+1]
            current_pos = 0
            for i in range(len(current_note_pitch)-1, -1, -1):
                if abs(current_note_pitch[i] - current_note[2]) >= 15.:
                    continue
                else:
                    current_pos = i
                    break
            # Then check the next note
            next_note_pitch = full_pitch[next_note[0]:next_note[1]+1]
            next_pos = 0
            for i in range(len(next_note_pitch)):
                if abs(next_note_pitch[i] - next_note[2]) >= 15.:
                    continue
                else:
                    next_pos = i
                    break
            # Considering that if the next note's start has decorative sounds, the first point in the histogram may be far
            # This might be problematic
            # So if this point is too far, then don't adjust, continue scanning
            if next_pos > 15:
                continue
            # Adjust the onset and offset to the midpoint of these two positions
            middle = int((current_note[0]+current_pos + next_note[0]+next_pos)/2)
            note[cnt][1] = middle-1
            note[cnt+1][0] = middle+1
    return note

def grace_note_detection(note, full_pitch):
    grace_note = []
    # First check each pair of adjacent notes, the offset and onset of adjacent notes differ by no more than 3
    # If the previous note is less than 15 long and the pitch difference is within 220cent, count it
    # If it is greater than 15 and less than min(25, half of the next note) and the pitch difference is within 220, also count it
    for i in range(0, len(note)-1):
        if note[i+1][0] - note[i][1] <= 3 and 10 < note[i][1] - note[i][0] < 15 and 70 < note[i+1][2] - note[i][2] < 220:
            grace_note.append(note[i])
        elif note[i+1][0] - note[i][1] <= 3 and 70 < note[i+1][2] - note[i][2] < 220 and 15 < note[i][1] - note[i][0] < min((note[i+1][1] - note[i+1][0])/2, 25) < 100:
            grace_note.append(note[i])
    # Then check individual notes
    for single_note in note:
        current_pitch = full_pitch[single_note[0]:single_note[1]+1]
        # Scan from left to right, count troughs and peaks in the area where the pitch is less than the note pitch, find the first peak and trough, and require both to be less than the note pitch
        # Also remove the leftmost 5%
        peak = []
        pitch_min = current_pitch[len(current_pitch)/20]
        pitch_max = 0
        pitch_max_pos = 0
        for i in range(len(current_pitch)-1, 5, -1):
            # Find the last point equal to pitch note from right to left
            if single_note[2] - 15 < current_pitch[i] < single_note[2] + 15:  # 30 is the size of the histogram bin
                pitch_max = current_pitch[i]
                pitch_max_pos = i
        bottom = 0
        top = 0
        for i in range(len(current_pitch)/20, pitch_max_pos):
            if single_note[2] > pitch_min > current_pitch[i]:
                bottom = i + single_note[0]
                pitch_min = current_pitch[i]
            elif pitch_max < current_pitch[i] < single_note[2]:
                top = i + single_note[0]
                pitch_max = current_pitch[i]
            elif current_pitch[i] > single_note[2]:
                break
        if min(bottom, top) > 0:
            # Have two
            peak.append(min(bottom, top))
            peak.append(max(bottom, top))
        elif max(bottom, top) > 0:
            # Have one
            peak.append(max(bottom, top))
        else:
            continue
        if 1 <= len(peak) <= 2:
            if single_note[2] - histogram_mean(single_note[0], single_note[0] + pitch_max_pos, full_pitch) > 90 and pitch_max_pos > 10:
                # The area from onset to just less than pitch note is decorative sound
                # The pitch of decorative sound is still determined by histogram
                grace_note.append([single_note[0], single_note[0] + int(0.9 * pitch_max_pos), histogram_mean(single_note[0], single_note[0] + int(0.9 * pitch_max_pos), full_pitch)])

    return grace_note

def vibrato_detection(note, full_pitch):
    vibrato = []

    for note_num in range(len(note)):
        single_note = note[note_num]

        # Notes that are too short are not considered
        if single_note[1] - single_note[0] < 50:
            continue
        current_pitch = full_pitch[single_note[0]:single_note[1]+1]

        # First subtract the pitch of this note from each point's pitch
        for i in range(len(current_pitch)):
            current_pitch[i] -= single_note[2]

        # Then find all areas where pitch is greater than 30cent and less than 150cent
        # The reason for choosing 30 and 150 is that the literature says the range of vibrato might be 30-150cent
        vibrato_possible = [False] * len(current_pitch)
        for i in range(len(current_pitch)):
            if 30. < abs(current_pitch[i]) < 150.:
                vibrato_possible[i] = True

        # Then use autocorrelation to find the periodicity within the current note
        # This algorithm is completely YIN's algorithm now
        # In the points with a periodicity of 4-9Hz, see if they meet the above conditions
        # If they do, and there are at least 4 peaks in this period region, then it is considered that this note contains vibrato

        # Set lag as 1-30, calculate the autocorrelation function value of the entire pitch
        # Here only calculate the autocorrelation of points that meet the above conditions (within the 30-150cent area)
        # Other points' pitch are considered as 0
        for i in range(len(current_pitch)):
            if not vibrato_possible:
                current_pitch[i] = 0

        acf_diff = []
        for lag in range(15, len(current_pitch)/2-1):
            temp = 0.
            for i in range(0, len(current_pitch)/2):
                temp += np.power((current_pitch[i] - current_pitch[i+lag]), 2)
            acf_diff.append(temp)

        # Should select the smallest value corresponding to the lag
        # The premise is that the acf value at this time is less than the threshold
        # Here take the threshold as 15w
        if min(acf_diff) < 200000:
            # This section may be a vibrato zone or stable zone, need to judge
            possible_lag = 15 + acf_diff.index(min(acf_diff))
            #Find all the maximum values and see if there are some subsequences of maximum values whose differences are all around lag.
            # If you find 3, it will be considered vibrato.

            # Find the crest
            peak = []
            for i in range(1, len(current_pitch)-1):
                if current_pitch[i-1] < current_pitch[i] > current_pitch[i+1]:
                    peak.append(i)
            # Find all the local maxima, see if there is a subsequence of some maxima with a common difference around the calculated period (within 5)
            # The method used is to start scanning from the beginning, see if there is a new peak at each element + lag position, if not, rescan
            pos = 0
            length = len(peak)
            vibrato_peak = []
            while pos < length:
                if len(vibrato_peak) == 0:
                    vibrato_peak.append(peak[pos])
                else:
                    flag = False
                    for deviation in range(-5, 6):
                        if vibrato_peak[-1] + possible_lag + deviation in peak:
                            vibrato_peak.append(vibrato_peak[-1] + possible_lag + deviation)
                            flag = True
                            break
                    if not flag:
                        # No more, see if the length exceeds 3
                        if len(vibrato_peak) >= 3:
                            # Is vibrato
                            vibrato.append([single_note[0] + vibrato_peak[0], single_note[0] + vibrato_peak[peaks]])
                            # Recalculate pitch, set it to the average of the entire section
                            note[note_num][2] = np.mean(full_pitch[single_note[0] + vibrato_peak[0]:single_note[0] + vibrato_peak[peaks]+1])
                            vibrato_peak = []
                            break
                        else:
                            # The peak region starting from pos is not vibrato, continue to search the subsequent peak region
                            pos += 1
                            vibrato_peak = []
                            break

                    # See if vibrato has been detected, if so, no need to continue
                    if len(vibrato) > 0 and vibrato[-1][0] > single_note[0]:
                        break
    return vibrato, note