#import play_sound_files# import play_audio
import time
import numpy as np
import os
import sounddevice
import vlc

def play_audio(fname, audio, player,media, wait=False):
    if True:
        path = os.getcwd() + '/../sound_files/'
        print(path)


        #p = vlc.MediaPlayer(path + fname + '.mp3')
        player.set_media(media)
        player.play()
        #p.play()
        print(fname)
        #print(dir(vlc), dir(p))

    if wait:
        time.sleep(0.5)
        while p.get_state() == vlc.State.Playing:
            pass

def main_call(audio, player,media):
    print("here")
    texts = 'Stop'
    play_audio(texts, audio, player,media)
    print("Playing audio")
    time.sleep(10)
    print("audio thread done")
