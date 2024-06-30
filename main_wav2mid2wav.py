
import os
import glob
import argparse
from timeit import default_timer as timer
from pydub import AudioSegment
import subprocess
import pretty_midi
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", default="assets/splices_audio_BMI/", help="Path to input audio file.")
    parser.add_argument("--folder", default="C:\\Users\\buddy\\Documents\\BeatsOn\\Voice-To-Midi\\hummed\\", help="Path to input audio file.")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo of the track in BPM.")
    parser.add_argument("--smooth", type=float, default=0.25,
                        help="Smooth the pitch sequence with a median filter "
                             "of the provided duration (in seconds).")
    parser.add_argument("--minduration", type=float, default=0.1,
                        help="Minimum allowed duration for note (in seconds). "
                             "Shorter notes will be removed.")
    parser.add_argument("--jams", action="store_const", const=True,
                        default=False, help="Also save output in JAMS format.")

    args = parser.parse_args()

    init_time = timer()

    FOLDER = args.folder + "*"

    IFS = ".mp3"  # delimiter
    IFS_CONVERT = ".mp3"
    BPM = args.bpm
    size_audio_files = 12  # in seconds

    subdirs = glob.glob(FOLDER)
    print(FOLDER, subdirs)



    subdirs_final = []
    for sub in subdirs:
        sub_folder = sub.split('/')[-1]
        #print(sub_folder)
        files = glob.glob(sub+"/*{}".format(IFS))
        for f in files:
            f_base = f.split(IFS)[0]
            filename = f_base.split('\\')[-1]
            f_out = f.replace(IFS, IFS_CONVERT)

            # wav to mid
            mid_dir = '{}_mid\\'.format(sub)
            if not os.path.exists(mid_dir):
                os.makedirs(mid_dir)

            command_str = 'python C:\\Users\\buddy\\Documents\\BeatsOn\\Voice-To-Midi\\audio_to_midi_melodia.py {f_base}{ifs} {mid_dir}{filename}.mid {bpm} --smooth ' \
                          '{smooth} --minduration {mindur}'.format(f_base=f_base, bpm=BPM, ifs=IFS,
                                                                   smooth=args.smooth, mindur=args.minduration,
                                                                   mid_dir=mid_dir, filename=filename)
            if args.jams:
                command_str += ' --jams'
            os.system(command_str)

            # mid to wav
            rec_dir = '{}_rec\\'.format(sub)
            if not os.path.exists(rec_dir):
                os.mkdir(rec_dir)
            os.system('timidity {mid_dir}{filename}.mid -Ow -o {rec_dir}{filename}_rec.wav'.
                      format(f_base=f_base, filename=filename, rec_dir=rec_dir, mid_dir=mid_dir))
            
            #mid to png
            #midi_to_piano_roll_image("{mid_dir}{filename}.mid".format(filename=filename, rec_dir=rec_dir, mid_dir=mid_dir)
                                     #,"{rec_dir}{filename}.png".format(filename=filename, rec_dir=rec_dir, mid_dir=mid_dir))                                     


    subdirs = subdirs_final  # [args.folder+'audio_16000_c1_16bits_music']


    end_time = timer()
    print("Program took {} minutes".format((end_time-init_time)/60))  # COGNIMUSE: 47 minutes
