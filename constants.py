import os
import numpy as np

TOKEN_URL = "https://accounts.spotify.com/api/token"

# Scopes provide Spotify users using third-party apps the confidence that only the information they choose to share will be shared.
SCOPE = "user-read-currently-playing user-read-playback-state user-modify-playback-state playlist-read-private playlist-modify-private user-library-modify user-library-read"

THRESHOLD = 0.900
ACTIONS = np.array(["play", "pause", "increase", "decrease", "next", "previous"])

# DATA_PATH = ...