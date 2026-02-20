import sys, os, subprocess, shutil, torch, librosa, numpy as np, time
from pathlib import Path
import scipy.signal 

# Force CPU Stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QTextEdit, QProgressBar, QHBoxLayout)
from PySide6.QtCore import QThread, Signal
from resemblyzer import VoiceEncoder, preprocess_wav

class ProcessingThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(str)

    def __init__(self, ref_file, big_folder, out_folder):
        super().__init__()
        self.ref_file = str(Path(ref_file).resolve())
        self.big_folder = str(Path(big_folder).resolve())
        self.out_folder = str(Path(out_folder).resolve())

    def run(self):
        try:
            run_id = int(time.time())
            temp_dir = Path(os.getcwd()) / f"temp_{run_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            encoder = VoiceEncoder(device="cpu")

            # 1. PROCESS THE SINGLE REFERENCE FILE FIRST
            self.log_signal.emit(f"🎵 Analyzing Reference: {Path(self.ref_file).name}")
            cmd_ref = f'demucs --two-stems=vocals --mp3 -o "{temp_dir}" -n htdemucs -d cpu "{self.ref_file}"'
            subprocess.run(cmd_ref, shell=True, capture_output=True)

            ref_name = Path(self.ref_file).stem
            ref_vocal_path = temp_dir / "htdemucs" / ref_name / "vocals.mp3"
            
            if not ref_vocal_path.exists():
                self.log_signal.emit("❌ Error: Could not process reference vocals.")
                return

            ref_duration = librosa.get_duration(path=str(ref_vocal_path))
            wav_ref = preprocess_wav(ref_vocal_path)
            ref_embed = encoder.embed_utterance(wav_ref)

            # 2. LOOP THROUGH THE FOLDER OF BIG VIDEOS
            big_files = [f for f in os.listdir(self.big_folder) if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi'))]
            
            for i, big_file in enumerate(big_files):
                big_path = str(Path(self.big_folder) / big_file)
                big_stem = Path(big_file).stem
                self.log_signal.emit(f"🎬 Scanning Video: {big_file}")

                # Isolate Big File Vocals
                cmd_big = f'demucs --two-stems=vocals --mp3 -o "{temp_dir}" -n htdemucs -d cpu "{big_path}"'
                subprocess.run(cmd_big, shell=True, capture_output=True)

                big_vocal_path = temp_dir / "htdemucs" / big_stem / "vocals.mp3"
                if not big_vocal_path.exists(): continue

                # Fingerprint the Big File
                wav_big = preprocess_wav(big_vocal_path)
                _, big_embeds, _ = encoder.embed_utterance(wav_big, return_partials=True, rate=10)
                
                # Match
                scores = np.inner(ref_embed, big_embeds)
                peaks, _ = scipy.signal.find_peaks(scores, height=0.70, distance=int(ref_duration * 10))

                if len(peaks) == 0:
                    self.log_signal.emit(f"   ⚠️ No matches in this video.")
                    continue

                self.log_signal.emit(f"   ✅ Found {len(peaks)} matches. Exporting...")

                for p_idx, peak_sample in enumerate(peaks):
                    start_time = max(0, (peak_sample * 0.1) - 0.1) # 0.1s early to avoid cut-off
                    output_name = f"MATCH_{big_stem}_{p_idx+1}.mp4"
                    output_path = os.path.join(self.out_folder, output_name)
                    
                    # --- COMPATIBILITY-FIRST FFMPEG COMMAND ---
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time),
                        "-t", str(ref_duration + 0.2),
                        "-i", big_path,
                        "-c:v", "libx264",        # Standard H.264
                        "-pix_fmt", "yuv420p",    # Most compatible pixel format for Windows
                        "-preset", "veryfast",    # Speed up encoding
                        "-c:a", "aac",            # Standard AAC audio
                        "-b:a", "192k",
                        output_path
                    ]
                    
                    subprocess.run(ffmpeg_cmd, capture_output=True)
                
                self.progress_signal.emit(int(((i + 1) / len(big_files)) * 100))

            self.finished_signal.emit("🏆 Finished! All clips are Windows-compatible.")
            
        except Exception as e:
            self.finished_signal.emit(f"❌ CRITICAL ERROR: {str(e)}")

class VoiceClipperGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Multi-Video Matcher")
        self.setMinimumWidth(650)
        self.ref_path = ""
        self.big_dir = ""
        self.out_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        h1, h2, h3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()

        self.lbl_ref = QLabel("<b>1.</b> Select ONE Reference File")
        btn_ref = QPushButton("Select File")
        btn_ref.clicked.connect(self.select_ref)
        h1.addWidget(self.lbl_ref, 1); h1.addWidget(btn_ref)

        self.lbl_big = QLabel("<b>2.</b> Select Folder of Big Videos")
        btn_big = QPushButton("Select Folder")
        btn_big.clicked.connect(self.select_big)
        h2.addWidget(self.lbl_big, 1); h2.addWidget(btn_big)

        self.lbl_out = QLabel(f"<b>3.</b> Saving to: /output")
        btn_out = QPushButton("Change Destination")
        btn_out.clicked.connect(self.select_out)
        h3.addWidget(self.lbl_out, 1); h3.addWidget(btn_out)

        self.btn_run = QPushButton("START GLOBAL SEARCH")
        self.btn_run.setStyleSheet("background-color: #27ae60; color: white; height: 50px; font-weight: bold;")
        self.btn_run.clicked.connect(self.start)

        self.pbar = QProgressBar()
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-family: Consolas;")

        for item in [h1, h2, h3, self.btn_run, self.pbar, self.log]:
            if isinstance(item, QHBoxLayout): layout.addLayout(item)
            else: layout.addWidget(item)
        self.setLayout(layout)

    def select_ref(self):
        self.ref_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Audio/Video")
        if self.ref_path: self.lbl_ref.setText(f"REF: {os.path.basename(self.ref_path)}")
    def select_big(self):
        self.big_dir = QFileDialog.getExistingDirectory(self, "Select Big Videos Folder")
        if self.big_dir: self.lbl_big.setText(f"FOLDER: {os.path.basename(self.big_dir)}")
    def select_out(self):
        self.out_dir = QFileDialog.getExistingDirectory(self, "Output Folder")
        if self.out_dir: self.lbl_out.setText(f"OUTPUT: {os.path.basename(self.out_dir)}")

    def start(self):
        if not self.ref_path or not self.big_dir: return
        self.btn_run.setEnabled(False)
        self.worker = ProcessingThread(self.ref_path, self.big_dir, self.out_dir)
        self.worker.log_signal.connect(self.log.append)
        self.worker.progress_signal.connect(self.pbar.setValue)
        self.worker.finished_signal.connect(lambda m: [self.log.append(m), self.btn_run.setEnabled(True)])
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceClipperGUI(); window.show()
    sys.exit(app.exec())