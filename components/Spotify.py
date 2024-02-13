import os

import requests
from dotenv import load_dotenv
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from spotipy.oauth2 import SpotifyOAuth

from constants import SCOPE

load_dotenv()

class SpotifyAPI(QThread):
    # This class represents connection with Spotify Web API

    def __init__(self, token: str = "") -> None:
        """
        Spotify constructor.
        token -> represents the authorization for Spotify
        """
        super().__init__()
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_uri = "http://localhost:8888/spotify-api/callback/"
        self.token = token
        self.headers = {"Authorization": f"Bearer {self.token}"}

    change_token = pyqtSignal(str)
    change_msg = pyqtSignal(str)

    def get_token(self) -> str:
        # function makes authorization and set token
        
        self.change_msg.emit("Waiting...")
        try:
            oauth = SpotifyOAuth(client_id=self.client_id, client_secret=self.client_secret, redirect_uri=self.redirect_uri, scope=SCOPE, show_dialog=True)
            self.token = oauth.get_access_token(as_dict=False, check_cache=False)
            self.change_token.emit(self.token)
            self.change_msg.emit("Successful authentication")
            return self.token
        except Exception as e:
            print(f"Exception: {e}")
            self.change_msg.emit("Authentication failed, please try again.")
    
    def get_playback_state(self) -> dict:
        # function gets information about the user's current playback state, including track, progress and active device
        # returns dict containing information about user, track and device
        try:
            endpoint = "https://api.spotify.com/v1/me/player"
            response = requests.get(endpoint, headers=self.headers)
            info = response.json()
            return info
        except Exception as e:
            print(f"Exception: {e}")

    def start_playback(self, playlist_id: str) -> None:
        # function starts playing paused track
        try:
            endpoint = "https://api.spotify.com/v1/me/player/play"
            requests.put(endpoint, headers=self.headers, data={"context_uri": f"spotify:playlist:{playlist_id}", "offset":{"position":1}})
        except Exception as e:
            print(f"Exception: {e}")

    def pause_playback(self) -> None:
        # function pauses playing track
        try:
            endpoint = "https://api.spotify.com/v1/me/player/pause"
            requests.put(endpoint, headers=self.headers)
        except Exception as e:
            print(f"Exception: {e}")

    def change_playing_status(self, playlist_id: str) -> None:
        # function starts play or pauses music based on current status
        try:
            is_playing = self.get_playback_state()["is_playing"]
            
            if is_playing:
                self.pause_playback()
            else:
                self.start_playback(playlist_id)
        except Exception as e:
            print(f"Exception: {e}")
    
    def skip_to_previous(self) -> None:
        # function skip to previous track
        try:
            endpoint = "https://api.spotify.com/v1/me/player/previous"
            requests.post(endpoint, headers=self.headers)
        except Exception as e:
            print(f"Exception: {e}")

    def skip_to_next(self) -> None:
        # function skip to next track
        try:
            endpoint = "https://api.spotify.com/v1/me/player/next"
            requests.post(endpoint, headers=self.headers)
        except Exception as e:
            print(f"Exception: {e}")

    def increase_volume(self) -> None:
        # function increases current volume by a step
        
        step = 5
        try:
            current_volume = self.get_playback_state()["device"]["volume_percent"]
            volume = current_volume + step if current_volume + step < 100 else 100

            endpoint = "https://api.spotify.com/v1/me/player/volume?volume_percent="
            requests.put(f"{endpoint}{volume}", headers=self.headers)
        except Exception as e:
            print(f"Exception: {e}")

    def decrease_volume(self) -> None:
        # function decreases current volume by a step
        
        step = 5
        try:
            current_volume = self.get_playback_state()["device"]["volume_percent"]
            volume = current_volume - step if current_volume - step > 0 else 0

            endpoint = "https://api.spotify.com/v1/me/player/volume?volume_percent="
            requests.put(f"{endpoint}{volume}", headers=self.headers)
        except Exception as e:
            print(f"Exception: {e}")

    def gesture_action(self, gesture: str, playlist_id: str) -> None:
        # function makes action based on gesture name
        # returns detected gesture string name
        
        if gesture == "play":
            self.change_playing_status(playlist_id)
        elif gesture == "pause":
            self.change_playing_status(playlist_id)
        elif gesture == "next":
            self.skip_to_next()
        elif gesture == "previous":
            self.skip_to_previous()
        elif gesture == "increase":
            self.increase_volume()
        elif gesture == "decrease":
            self.decrease_volume()

    def run(self):
        self.get_token()
