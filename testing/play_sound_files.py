# Quick script to initialize audio files
#from gtts import gTTS
import sounddevice
import vlc
import time
import os

#make the text string from the goal value
cwd = os.getcwd()
save_dir = cwd + '/../sound_files/'
strings = ['Walk at a slow speed','Walk as if you were walking a small dog','Walk as if you were walking home after a really bad day','Walk as if you were walking through a park','Walk as if you were walking in the grocery store','Walk as if you were walking from the bedroom to the kitchen','Walk as if you were walking through a field','Walk as if you were walking home with a group of friends at night','Walk as if you were walking home from the bus after class','Walk as if you were walking home from class','Walk at your typical speed','Walk at your normal speed','Walk as if you were walking a big dog','Walk as if you were walking to class','Walk as if you were going to pick up an item off of a table','Walk as if you were walking across the street','Walk as if you were walking home alone at night','Walk as if you were jay-walking','Walk as if you were walking to catch a bus', 'Wlk as if you were walking to class and you were late', 'Walk as fast as possible']
strings_speeds = [0.91, 1.17, 1.21, 1.23, 1.22, 1.27, 1.28, 1.281, 1.37, 1.371, 1.38, 1.381, 1.45, 1.46, 1.5, 1.56, 1.57, 1.82, 1.84, 1.9, 2.16]
print(save_dir)



def play_audio(fname, audio, wait=False):
    if True:
        path = os.getcwd() + '/../sound_files/'
        print(path)
        try:
            p = vlc.MediaPlayer(path + fname + '.mp3')
            p.play()
            print(fname)
            print(dir(vlc), dir(p))
        except:
            print("error playing")

    if wait:
        time.sleep(0.5)
        while p.get_state() == vlc.State.Playing:
            pass

def play_audio2(fname, wait=False):
    if True:
        path = os.getcwd() + '/../sound_files/'
        try:
            p = vlc.MediaPlayer(path + fname + '.mp3')
            p.play()
            print(fname)
        except:
            print("error playing")

    if wait:
        time.sleep(0.5)
        while p.get_state() == vlc.State.Playing:
            pass

#for i in range(len(strings)):
# i=-1
# string_path = save_dir + strings[i] + '.mp3'
# print(string_path)
# play_audio(strings[i])
# print(i)
# time.sleep(5)
# p = vlc.MediaPlayer(string_path)
# p.play()
# time.sleep(0.5)
# while p.get_state() == vlc.State.Playing:
#     pass
# print('0')
    #make_mp3_from_string(strings[i], save_dir+strings[i] + '.mp3')

def make_mp3_goals():
    goals = list(range(-90, 100, 10))
    for goal in goals:
        text_string = build_string(goal)
        fname = str(goal) + ".mp3"
        make_mp3_from_string(text_string)



def check_sound_devices():
    devs = sounddevice.query_devices()
    print(devs) # Shows current output and input as well with "<" abd ">" tokens

def test_wait_for_finish():
    p = vlc.MediaPlayer("turn_audio_files/90.mp3")
    p.play()
    time.sleep(0.5)
    while p.get_state() == vlc.State.Playing:
        pass
    time.sleep(0.5)
    p = vlc.MediaPlayer("turn_audio_files/-30.mp3")
    p.play()
