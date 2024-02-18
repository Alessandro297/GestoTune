<img src="logos_graphics/gestotune_logo.png" alt="Image" width="100" height="100">  

# GestoTune

## Description

GestoTune is an application that allows to control spotify playlists using hand gestures. The application is able to recognize enrolled users and associate to them their prefered playlist.
To do this, **Python 3.11.5** was used with **dlib**, **OpenCV** and **MediaPipe** libraries. **Pytorch** model was trained on custom dataset, which was created by modifying the subsample of the **HaGRID** dataset.

## How to install and run

**Remember: to use this app you need to have premium version of Spotify.**

1. Download repository
   ```
   git clone https://github.com/kzaleskaa/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```
2. Create a new project at [Spotify Dashboard](https://developer.spotify.com/dashboard/) and edit settings - add `http://localhost:8888/spotify-api/callback/` in Redirect URLs.
3. Create your environment and activate it
   ```bash
   $ python -m venv venv
   ```
4. Install requirements
   ```bash
   $ pip install -r .\requirements.txt
   ```
5. In root project directory, create .env file with following content (from your Spotify project Dashboard):
   ```
   CLIENT_ID= ...
   CLIENT_SECRET= ...
   ```
6. Start app
   ```
   python main.py
   ```
7. Click on `login` to connect to spotify API
8. If using the app for the first time, procede with `enroll`, otherwise with `start`.
9. Open your spotify app, start play music and use this app to control it.

## Gestures
![legenda](https://github.com/Alessandro297/GestoTune/assets/152632307/08137d0e-a168-4665-b1d7-58fa7a1f350b)
<div align="center">
   
## Spotify API

> Based on simple REST principles, the Spotify Web API endpoints return JSON metadata about music artists, albums, and tracks, directly from the Spotify Data Catalogue.

If no action on Spotify is made, please open your app and start play music manually. Then, you can use this app to control it.


