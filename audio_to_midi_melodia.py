# CREATED: 11/9/15 3:57 PM by Justin Salamon <justin.salamon@nyu.edu>

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
from utils import *

'''
Extract the melody from an audio file and convert it to MIDI.

The script extracts the melody from an audio file using the Melodia algorithm,
and then segments the continuous pitch sequence into a series of quantized
notes, and exports to MIDI using the provided BPM. If the --jams option is
specified the script will also save the output as a JAMS file. Note that the
JAMS file uses the original note onset/offset times estimated by the algorithm
and ignores the provided BPM value.

Note: Melodia can work pretty well and is the result of several years of
research. The note segmentation/quantization code was hacked in about 30
minutes. Proceed at your own risk... :)

usage: audio_to_midi_melodia.py [-h] [--smooth SMOOTH]
                                [--minduration MINDURATION] [--jams]
                                infile outfile bpm


Examples:
python audio_to_midi_melodia.py --smooth 0.25 --minduration 0.1 --jams
                                ~/song.wav ~/song.mid 60
'''


def midi_to_notes(midi, fs, hop, smooth, minduration):

    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi
    # print(len(midi),len(midi_filt))

    notes = []
    p_prev = None
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p is None:
            continue  # Skip this iteration if p is None
        if p_prev is None:  # Ensure p_prev has a default value if None
            p_prev = p  # You might adjust this logic depending on your needs
        
        if p == p_prev:
            duration += 1
        else:
            if p_prev > 0:  # This is where your error occurs
                duration_sec = duration * hop / float(fs)
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes

def filter_anomalies_and_isolation(notes, deviation_threshold_multiplier=2, skip_threshold_multiplier=2, isolation_threshold_seconds=0.5):
    # Calculate the median pitch and its deviations
    median_pitch = np.median([note[2] for note in notes])
    deviations = [abs(note[2] - median_pitch) for note in notes]
    median_deviation = np.median(deviations)
    deviation_threshold = median_deviation * deviation_threshold_multiplier
    
    potential_anomalies = set()
    for i, dev in enumerate(deviations):
        if dev > deviation_threshold:
            potential_anomalies.add(i)

    # Calculate the average non-zero pitch change for the skip threshold
    avg_change = np.mean([abs(notes[i + 1][2] - notes[i][2]) for i in range(len(notes) - 1)
                          if notes[i + 1][2] != notes[i][2] and i not in potential_anomalies])
    skip_threshold = avg_change * skip_threshold_multiplier

    # Check for temporal isolation from the nearest non-anomalous notes
    note_end_times = [note[0] + note[1] for note in notes]  # End times of each note
    confirmed_anomalies = set()
    for i in potential_anomalies:
        # Find the nearest non-anomalous previous note
        prev_index = i - 1
        while prev_index in potential_anomalies and prev_index > 0:
            prev_index -= 1

        # Find the nearest non-anomalous next note
        next_index = i + 1
        while next_index in potential_anomalies and next_index < len(notes) - 1:
            next_index += 1

        is_isolated = True  # Assume the note is isolated until proven otherwise

        # Check isolation with previous non-anomalous note
        if prev_index >= 0:
            prev_note_gap = notes[i][0] - note_end_times[prev_index]
            if prev_note_gap <= isolation_threshold_seconds:
                is_isolated = False
        
        # Check isolation with next non-anomalous note
        if next_index < len(notes):
            next_note_gap = notes[next_index][0] - note_end_times[i]
            if next_note_gap <= isolation_threshold_seconds:
                is_isolated = False

        # Confirm the anomaly if isolated and skip conditions are met
        if is_isolated:
            if i > 0 and abs(notes[i][2] - notes[prev_index][2]) > skip_threshold:
                confirmed_anomalies.add(i)
            elif next_index < len(notes) and abs(notes[next_index][2] - notes[i][2]) > skip_threshold:
                confirmed_anomalies.add(i)
    print("Number of potential anomalies:", len(potential_anomalies))
    print("Number of confirmed anomalies:", len(confirmed_anomalies))
    # Filter out confirmed anomalies to get the final list of notes
    filtered_notes = [note for i, note in enumerate(notes) if i not in confirmed_anomalies]
    return filtered_notes

def audio_to_midi_melodia(infile, outfile, bpm, smooth=0.25, minduration=0.1,
                          savejams=False):

    # define analysis parameters
    fs = 44100
    hop_length = 128  # Librosa's default hop length is 512

    # load audio using librosa
    print("\n \n \n Loading audio...")
    data, sr = librosa.load(infile, sr=fs)  # librosa ensures audio is mono
    print("Extracting melody f0 with MELODIA...")
    melody = vamp.collect(data, sr, "mtg-melodia:melodia",
                          parameters={"voicing": 0.2})

    # hop = melody['vector'][0]
    pitch = melody['vector'][1]

    # impute missing 0's to compensate for starting timestamp
    pitch = np.insert(pitch, 0, [0]*8)

    timestamps = 8 * 128/44100.0 + np.arange(len(pitch)) * (128/44100.0)
    #print(pitch)
    plt.figure(figsize=(10,4))
    plt.plot(timestamps,pitch)
    plt.savefig("{outfile}pitch.png".format(outfile=outfile))
    plt.close()
    print("Converting Hz to MIDI notes...")
    midi_pitch = hz2midi(pitch)  # Use the same hz2midi function
    cent_pitch = freq2cent(pitch)
    #print(midi_pitch)
    plt.figure(figsize=(10,4))
    plt.plot(timestamps,midi_pitch)
    plt.savefig("{outfile}pitchmidi.png".format(outfile=outfile))
    plt.close()
    # segment sequence into individual midi notes
    notes_prefiltered = midi_to_notes(cent_pitch, fs, hop_length, smooth, minduration)
    notes = convert_to_onset_offset_pitch(notes_prefiltered,hop_length,fs)

    notes = note_postprocessing(notes, midi_pitch)
    vibrato, notes = vibrato_detection(notes, midi_pitch)
    notes = small_note_segment(notes, midi_pitch, vibrato)
    notes = onset_offset_adjust(notes, midi_pitch)
    notes = convert_to_onset_duration_pprev(notes,hop_length,fs)
    #notes = filter_anomalies_and_isolation(notes, deviation_threshold_multiplier=2, skip_threshold_multiplier=2, isolation_threshold_seconds=0.1)

    # save note sequence to a midi file
    print("Saving MIDI to disk...")
    print(notes)
    save_midi(outfile, notes, bpm)
    #save_midi(outfile[:-4]+"prefiltered.mid",notes_prefiltered,bpm)

    if savejams:
        print("Saving JAMS to disk...")
        jamsfile = outfile.replace(".mid", ".jams")
        track_duration = len(data) / float(fs)
        save_jams(jamsfile, notes, track_duration, os.path.basename(infile))

    print("Conversion complete.")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", default="assets/10.wav", help="Path to input audio file.")
    parser.add_argument("outfile", default="assets/10.mid", help="Path for saving output MIDI file.")
    parser.add_argument("bpm", type=int, default=146, help="Tempo of the track in BPM.")
    parser.add_argument("--smooth", type=float, default=0.22,
                        help="Smooth the pitch sequence with a median filter "
                             "of the provided duration (in seconds).")
    parser.add_argument("--minduration", type=float, default=0.1,
                        help="Minimum allowed duration for note (in seconds). "
                             "Shorter notes will be removed.")
    parser.add_argument("--jams", action="store_const", const=True,
                        default=False, help="Also save output in JAMS format.")

    args = parser.parse_args()

    audio_to_midi_melodia(args.infile, args.outfile, args.bpm,
                          smooth=args.smooth, minduration=args.minduration,
                          savejams=args.jams)
