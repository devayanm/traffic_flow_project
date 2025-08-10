# 🚦 Traffic Flow Analysis using YOLOv8 and Centroid Tracking

## 📌 Overview

This project is a **Traffic Flow Analysis System** developed as part of a recruitment assignment.  
It detects and counts vehicles in each lane of a given traffic video using **YOLOv8 object detection** and a **Centroid Tracking algorithm** to avoid duplicate counting.

The program processes a video frame-by-frame, assigns a unique ID to each detected vehicle, determines which lane it belongs to, and updates the counts.  
Two main outputs are generated:

1. **Annotated Video** – showing detected vehicles, lane markings, and live counts.
2. **CSV Report** – with details for every detected vehicle.

---

## 🎯 Features

- **Automatic Video Download** from YouTube (via `yt-dlp`) or use of a local file.
- **Vehicle Detection** using YOLOv8 pre-trained on COCO dataset.
- **Centroid Tracking** to maintain unique vehicle IDs across frames.
- **Lane Assignment** for accurate per-lane counting.
- **CSV Output** with detailed statistics:
  - Vehicle ID
  - Lane Number
  - Frames Seen
  - First Frame
  - First Timestamp (seconds)
- Modular and easily adaptable for:
  - Different lane configurations
  - Other object classes
  - Live camera feeds

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Libraries:**
  - `opencv-python-headless` – Image and video processing
  - `numpy` – Numerical operations
  - `ultralytics` – YOLOv8 detection
  - `pytube` – YouTube video download (fallback option)
  - `yt-dlp` – Faster/more reliable YouTube download
- **Model:** YOLOv8 Nano (`yolov8n.pt`)

---

## 📂 Project Structure

```

traffic_flow_project/
│
├── main.py                # Main script for processing
├── tracker.py             # Centroid Tracker implementation
├── config.json            # Configurable parameters
├── run.sh                 # Script to automate setup & run
├── requirements.txt       # Python dependencies
├── output/
│   ├── annotated.mp4      # Annotated output video
│   ├── vehicle_counts.csv # Vehicle data report
└── README.md              # Project documentation

```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/devayanm/traffic_flow_project.git
cd traffic-flow-project
```

### 2️⃣ Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Run with YouTube Video Download

```bash
./run.sh
```

This will:

- Download the traffic video from YouTube.
- Process it using YOLOv8 + Centroid Tracking.
- Save results in `output/`.

### Option 2: Run with a Local Video

```bash
python main.py --input_video path/to/video.mp4
```

---

## ⚙️ Configuration

You can modify `config.json` to:

- Change YOLO model (`yolov8n.pt`, `yolov8s.pt`, etc.)
- Adjust confidence threshold
- Select target classes (car, truck, bus, motorcycle, etc.)
- Define custom lane polygons
- Set tracker parameters (`max_disappeared`, `max_distance`)

Example:

```json
{
  "model": "yolov8n.pt",
  "min_confidence": 0.35,
  "classes": ["car", "truck", "bus", "motorcycle"],
  "lanes": "equal_vertical_3",
  "max_disappeared": 30,
  "max_distance": 80,
  "output_dir": "output"
}
```

---

## 📊 Output Files

### 1️⃣ Annotated Video (`annotated.mp4`)

- Shows lane boundaries
- Displays live lane counts
- Marks each vehicle with a unique ID

### 2️⃣ CSV Report (`vehicle_counts.csv`)

| vehicle_id | lane | frames_seen | first_frame | first_timestamp_seconds |
| ---------- | ---- | ----------- | ----------- | ----------------------- |
| 1          | 2    | 150         | 12          | 0.40                    |
| 2          | 3    | 180         | 15          | 0.50                    |

---

## 📹 Demo Preview

![Traffic Flow Analysis Demo](assets/demo.gif)

🔗 [Full Annotated Video](https://drive.google.com/file/d/1QAljJxNlfne6RW_Lo9WaBnPkTSTzLTEW/view?usp=sharing)

---

## 🧠 Approach

1. **Video Acquisition**

   - Used `yt-dlp` to download the given traffic video.

2. **Detection**

   - Loaded YOLOv8 Nano model pre-trained on COCO dataset.
   - Filtered detections to relevant classes (cars, buses, trucks, motorcycles).

3. **Tracking**

   - Used Centroid Tracking to assign persistent IDs.
   - Tracked vehicle positions frame-by-frame.

4. **Lane Assignment**

   - Assigned vehicles to lanes based on their centroid positions.

5. **Counting & Logging**

   - Counted a vehicle only once upon first lane entry.
   - Wrote all data to CSV instantly to ensure synchronization with video.

6. **Output**

   - Generated annotated MP4 video and CSV report.

---

## 🏆 Challenges & Solutions

- **Slow Processing on CPU**
  YOLO inference took time for every frame (\~2.5 hours total).
  → Solution: Chose accuracy over speed for assignment. Could use frame skipping or resizing for faster runtime.
- **Lane Detection Accuracy**
  Vehicles on lane borders were sometimes misclassified.
  → Solution: Added fallback check using `x` coordinate when not inside any lane polygon.
- **Avoiding Duplicate Counts**
  → Solution: Used Centroid Tracker’s ID assignment to count vehicles only once.

---

## 🔮 Future Improvements

- Multi-threaded processing for faster execution.
- Use of YOLOv8 on GPU for real-time detection.
- Dynamic lane configuration via UI.
- Live feed integration from CCTV or IP cameras.

---

## 👤 Author

**Name:** Devayan Mandal
**Location:** India
**Email:** [devayan9689@gmail.com](devayan9689@gmail.com)

---

## 📜 License

This project is for recruitment assignment purposes only.
Do not distribute without permission.
