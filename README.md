# Traffic Flow Analysis — Skytel Assignment

**What this project does**
- Downloads the specified YouTube traffic video (optional) or processes a local video file.
- Detects vehicles using YOLOv8 (Ultralytics) pretrained COCO model.
- Tracks vehicles across frames using a simple centroid tracker to avoid duplicate counts.
- Counts vehicles separately in **3 lanes** (configurable).
- Produces:
  - `output/annotated.mp4` — annotated video with lane overlays, bounding boxes, IDs, and live counts.
  - `output/vehicle_counts.csv` — CSV with columns: `vehicle_id,lane,frames_seen,first_frame,first_timestamp`.
- Prints per-lane totals at the end.

**Contents**
- `main.py` — main processing script
- `tracker.py` — centroid-based tracker implementation
- `requirements.txt` — Python dependencies
- `config.json` — lane / video configuration
- `run.sh` — example run script
- `README.md` — this file

**Quick setup (Linux / WSL / macOS)**
1. Create virtualenv:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: `ultralytics` will download YOLOv8 weights on first run.
3. Run:
   ```bash
   # Option A: download the YouTube video automatically
   python main.py --download_youtube --youtube_url "https://www.youtube.com/watch?v=MNn9qKG2UFI"

   # Option B: use local video
   python main.py --input_video /path/to/video.mp4
   ```
4. Outputs will be in the `output/` folder.

**Notes**
- The script defines 3 equal-width vertical lanes by default. To customize lanes, edit `config.json`.
- This implementation uses a lightweight centroid tracker. For better accuracy in crowded scenes, replace tracker with DeepSORT.
- The demo video and GitHub repo must be produced by the submitter. This project provides the runnable code and instructions.

**Contact**
If you want, I can also:
- Generate a GitHub-friendly repo structure and commit messages
- Record a 1–2 minute demo video script and commands to create the demo
# traffic_flow_project
