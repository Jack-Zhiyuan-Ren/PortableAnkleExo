from multiprocessing import Process, Queue
import play_sound_files# import play_audio
import audio_caller as ac
import time
import play_sound_files# import play_audio
import vlc
import sounddevice
import os
instance = vlc.Instance()
player = instance.media_player_new()
path = os.getcwd() + '/../sound_files/'
source = path + 'Stop.mp3'
media = instance.media_new(source)
#player = instance.media_player_new()
audio = True
cma_thread = Process(target=ac.main_call, args=(audio,player,media,), daemon=True)
cma_thread.start()
print("started thread")
time.sleep(10)
print("ending thread")
#play_sound_files.play_audio('Stop',audio)
time.sleep(5)
