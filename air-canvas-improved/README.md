# Air Canvas Studio 🎨

A browser-based, touch-free drawing app. Draw with your index finger, erase with a fist, and pause with an open palm using your webcam.

## Features
- Real-time hand tracking with MediaPipe Tasks Vision
- Smooth, gap-resistant strokes using segment subdivision and adaptive fingertip smoothing
- Gesture stabilization to reduce accidental mode changes
- Fist-controlled eraser with a wide, rounded stroke
- Color palette and adjustable brush size
- Undo, clear, and export drawings as transparent PNG
- Minimal responsive interface
- Webcam processing stays in the browser

## Run locally
Camera access requires `localhost` or HTTPS.

1. Download or clone this repository.
2. Open the folder in VS Code.
3. Start a local server from the folder:

   ```bash
   python -m http.server 8000
   ```

4. Open http://localhost:8000
5. Click **Enable camera** and allow camera access.

You can also use the VS Code Live Server extension.

## Gestures
- **Index finger only:** draw
- **Fist:** erase
- **Open palm:** pause drawing

Gesture recognition can vary with lighting, camera angle, and hand visibility.

## Tech stack
- HTML5
- CSS3
- JavaScript ES modules
- Canvas 2D API
- MediaPipe Tasks Vision

## Notes
The hand-tracking package and model are loaded from their CDNs, so an internet connection is required on first load. No video is uploaded by this app.

## Author
Arya Desai

## Credits
Hand tracking is powered by MediaPipe. This is an independent implementation using standard browser APIs.
