# Finger Frame AI 🎬✨

**Try it:** [AI restyle](https://milleniumvietnam.github.io/finger-frame-effect-ai/)
· [live camera, no key](https://milleniumvietnam.github.io/finger-frame-effect-ai/live/)
(upstream: [sophiamyang.github.io](https://sophiamyang.github.io/finger-frame-effect-ai/))

Upload a video of the two-hand finger-frame gesture — get it back with an
**AI-generated world inside the frame**. The whole video is restyled by a
video-to-video model (motion, blinks, and all), then composited so the
finger frame acts as a window into the animated version.

## The finger-frame family

| App | Generation | Latency |
|---|---|---|
| [**`live/`**](live/) — live camera, local effects (in this repo) | Canvas 2D (Van Gogh, cartoon, neon, glitch, pixel, thermal) | none |
| **this app** — recorded video, AI restyle | Gemini Omni Flash (offline video edit) | minutes |
| [finger-frame-effect](https://sophiamyang.github.io/finger-frame-effect/) ([repo](https://github.com/sophiamyang/finger-frame-effect)) — the original live-camera app | Canvas 2D | none |
| [finger-frame-effect-lucy](https://sophiamyang.github.io/finger-frame-effect-lucy/) ([repo](https://github.com/sophiamyang/finger-frame-effect-lucy)) — live camera, live AI | Decart Lucy 2.5 (realtime video-to-video) | ~real time |

![Example: AI-animated world inside the finger frame](examples/final.gif)

*Real hands, AI world — generated with the default "3D animated movie"
style ([full-quality mp4](examples/final.mp4)).*

## How it works

1. **Restyle** — the uploaded video is sent to
   [Gemini Omni Flash video editing](https://ai.google.dev/gemini-api/docs/omni)
   with your chosen style (3D animated movie, anime, claymation,
   watercolor, or a custom prompt). This is a true video model: the whole
   clip is regenerated, so the animated version moves exactly like you.
   Every prompt gets a strict-alignment suffix appended — same framing, no
   zoom/crop/recentering, facial features at the same screen coordinates,
   expression preserved frame by frame (mouth openness, blinks, gaze) — so
   the result lines up behind the finger-frame window.
2. **Track** — MediaPipe Hand Landmarker finds both hands per frame, and the
   finger-frame quad is tracked with the same audited pipeline as the live
   app (anatomical corner ordering — crossing your fingers renders the
   bowtie — spread/area gates with hysteresis, teleport rejection,
   velocity-adaptive smoothing, dropout hold, presence fade).
3. **Composite** — the AI video is revealed through the tracked quad with
   the dashed marching-ants outline and pulsing corner dots.
4. **Export** — the result records to a downloadable video — MP4 where the
   browser supports recording it (Safari, newer Chrome), otherwise `.webm`
   (convert with `ffmpeg -i finger-frame-ai.webm -c:v libx264 out.mp4`).

## Live camera mode (`live/`)

No key, no upload, no waiting: open [`live/`](live/), start the camera, and
the finger frame becomes a window onto a restyled version of the room. Six
effects, all Canvas 2D and all local:

| | Effect | What it does |
|---|---|---|
| `1` | Van Gogh | short curved brush strokes laid along the image's edge directions, with a slow swirl field filling the flat areas |
| `2` | Cartoon | posterized colour with a Sobel ink outline |
| `3` | Neon | edges only, hue swept over time, bloomed with a blur pass |
| `4` | Glitch | RGB channel split, scanlines, chunky band displacement |
| `5` | Pixel art | nearest-neighbour downsample plus posterize |
| `6` | Thermal | luminance through a heat-map ramp |

Tracking is the same pipeline as the AI app above, so the window behaves
identically — anatomical corner ordering, spread/area gates with hysteresis,
teleport rejection, velocity-adaptive smoothing, dropout hold, presence fade.
Effects run on a downscaled copy of the window region (≤320px), which keeps
them real time regardless of how big the frame gets on screen.

Keys: `1`–`6` pick an effect, `←`/`→` cycle, `M` mirrors, `S` saves a PNG,
`R` starts/stops recording (MP4 where the browser supports it, otherwise
`.webm`). Nothing is uploaded — the camera stream never leaves the tab.

## Bring your own key

The AI step uses your own [Gemini API key](https://aistudio.google.com/apikey),
entered in the app. It stays in your browser (localStorage only if you check
"remember") and is sent only to Google's API. Generation is billed per
video and takes a few minutes. Keep clips under ~15MB (a few seconds of
720p — any common format: mp4, mov, webm); larger files exceed the inline
upload limit. No key? The **placeholder style** button runs the full
track-composite-export pipeline with a hue-shifted stand-in so you can try
everything for free.

## Run locally

Any static server works:

```bash
python3 -m http.server 8124
```

Then open http://localhost:8124 for the AI app, or
http://localhost:8124/live/ for the live-camera one. A `?src=<file>` query
param loads a video from the server directory (dev convenience).

## Publishing

`.github/workflows/pages.yml` publishes the repository to GitHub Pages on
every push to `main` — `/` serves the AI app and `/live/` the live-camera
one. It switches Pages on by itself the first time it runs
(`configure-pages` with `enablement: true`). If your account or org blocks
that, turn it on by hand instead — **Settings → Pages → Build and
deployment → Source: GitHub Actions** — and re-run the workflow; nothing
else in the repo depends on it.

## CLI alternative (Python)

The same pipeline as offline scripts — useful for batch work or
frame-accurate H.264 output:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export GEMINI_API_KEY=...
.venv/bin/python stylize.py input.mp4 -o stylized.mp4      # AI restyle
.venv/bin/python composite.py input.mp4 stylized.mp4 -o final.mp4
```

Any input format ffmpeg/OpenCV can read works (mp4, mov, webm, …).
`composite.py` needs `ffmpeg` on PATH, outputs H.264 MP4, and carries over
the original audio track when present.

## Notes

- Input/output media are gitignored — personal footage stays local. The
  only committed media is the sample under `examples/`.
