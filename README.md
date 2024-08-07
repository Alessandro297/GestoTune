# GestoTune

<img src="logos_graphics/gestotune_logo.png" alt="Image" width="150" height="150">  


## Description

GestoTune is an application that allows to control spotify playlists using hand gestures. The application is able to recognize enrolled users and associate to them their prefered playlist.
To do this, **Python 3.11.5** was used with **dlib**, **OpenCV**, **MediaPipe** and **PyQt5** libraries. **Pytorch** model was trained on custom dataset, which was created by modifying the subsample of the **HaGRID** dataset.
Note that the gesture-control part is meant to be used with the right hand, and does not detect the left one.

Demo videos are available via [this link](https://drive.google.com/drive/folders/1c8y3N0-ZQ73lF7QIIiLYRUeI46pH8Uw5?usp=sharing) in our Google Drive folder.

## How to install and run

**Remember: to use this app you need to have premium version of Spotify.**

1. Download repository
   ```
   git clone https://github.com/Alessandro297/GestoTune.git
   cd GestoTune
   ```
2. Create a new project at [Spotify Dashboard](https://developer.spotify.com/dashboard/) and edit settings - add `http://localhost:8888/spotify-api/callback/` in Redirect URLs and select "Web API" for the question asking which APIs are you planning to use.
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

## Extra files
In the folder `model' there's the best model, deployed in the application.
In the same folder is possible to find a notebook in which is shown how the dataset was created and other possible models which have been tested.
Via [this link](https://drive.google.com/drive/folders/1c8y3N0-ZQ73lF7QIIiLYRUeI46pH8Uw5?usp=sharing
) one can access to our Google Drive folder which contains the already processed images, as well as the extracted landmarks and labels of the final dataset.

## Spotify API

> Based on simple REST principles, the Spotify Web API endpoints return JSON metadata about music artists, albums, and tracks, directly from the Spotify Data Catalogue.

If no action on Spotify is made, please open your app and start play music manually. Then, you can use this app to control it.


