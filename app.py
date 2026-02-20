import sys, os, subprocess, shutil, torch, librosa, numpy as np, time
from pathlib import Path
import scipy.signal 

# Force CPU Stability
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

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

            # 1. ANALYZE REFERENCE DURATION
            self.log_signal.emit(f"🎵 Pattern Analysis...")
            cmd_ref = f'demucs --two-stems=vocals --mp3 -o "{temp_dir}" -n htdemucs -d cpu "{self.ref_file}"'
            subprocess.run(cmd_ref, shell=True, capture_output=True)

            ref_name = Path(self.ref_file).stem
            ref_vocal_path = temp_dir / "htdemucs" / ref_name / "vocals.mp3"
            
            # GET STRICT REF DURATION (e.g., 4.5s)
            ref_duration = librosa.get_duration(path=str(ref_vocal_path))
            self.log_signal.emit(f"📑 Reference phrase length: {ref_duration:.2f}s")

            wav_ref = preprocess_wav(ref_vocal_path)
            ref_embed = encoder.embed_utterance(wav_ref)

            # 2. PROCESS BIG VIDEOS
            big_files = [f for f in os.listdir(self.big_folder) if f.lower().endswith(('.mp4', '.mov', '.mkv'))]
            
            for i, big_file in enumerate(big_files):
                big_path = str(Path(self.big_folder) / big_file)
                big_stem = Path(big_file).stem
                self.log_signal.emit(f"🎬 Processing: {big_file}")

                cmd_big = f'demucs --two-stems=vocals --mp3 -o "{temp_dir}" -n htdemucs -d cpu "{big_path}"'
                subprocess.run(cmd_big, shell=True, capture_output=True)

                big_vocal_path = temp_dir / "htdemucs" / big_stem / "vocals.mp3"
                if not big_vocal_path.exists(): continue

                y_vocal, sr = librosa.load(str(big_vocal_path), sr=16000)
                wav_big = preprocess_wav(big_vocal_path)
                _, big_embeds, _ = encoder.embed_utterance(wav_big, return_partials=True, rate=10)
                
                scores = np.inner(ref_embed, big_embeds)
                
                # INNOVATION: DISTANCE LOCK
                # We set 'distance' to roughly the length of the reference file.
                # If ref is 4.5s, distance is 45 samples (at 10Hz).
                # This prevents cutting the same phrase multiple times.
                min_dist = int(ref_duration * 10 * 0.8) # 80% of ref duration
                peaks, _ = scipy.signal.find_peaks(scores, height=0.75, distance=min_dist)

                for p_idx, peak_sample in enumerate(peaks):
                    # Initial AI Match Point
                    raw_start = max(0, (peak_sample * 0.1) - 0.2)
                    
                    # --- VOICE LOWERING REFINEMENT (+/- 0.5s) ---
                    # Look at the audio 0.5s before and after the theoretical end
                    theo_end = raw_start + ref_duration
                    search_start = int(max(0, theo_end - 0.5) * sr)
                    search_end = int((theo_end + 0.5) * sr)
                    end_window = y_vocal[search_start:search_end]
                    
                    # Detect where the sound actually drops in this 1-second window
                    # This finds the "Natural Pause" near our reference time
                    intervals = librosa.effects.split(end_window, top_db=25)
                    if len(intervals) > 0:
                        # Find the end of the last sound in the window
                        refined_end_offset = intervals[-1][1] / sr
                        final_duration = (theo_end - 0.5 - raw_start) + refined_end_offset
                    else:
                        final_duration = ref_duration

                    # --- EXPORT ---
                    output_name = f"DANCE_{big_stem}_{p_idx+1}.mp4"
                    output_path = os.path.join(self.out_folder, output_name)
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", big_path,
                        "-ss", str(raw_start),
                        "-t", str(final_duration + 0.3), # Tiny buffer for pose completion
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-c:a", "aac", "-avoid_negative_ts", "make_zero",
                        output_path
                    ]
                    
                    subprocess.run(ffmpeg_cmd, capture_output=True)
                    self.log_signal.emit(f"      ✅ Phrase Exported: {final_duration:.2f}s")

                self.progress_signal.emit(int(((i + 1) / len(big_files)) * 100))

            shutil.rmtree(temp_dir)
            self.finished_signal.emit("🏆 High-Accuracy Dataset Complete!")
            
        except Exception as e:
            self.finished_signal.emit(f"❌ Error: {str(e)}")

# (The VoiceClipperGUI class remains the same)

# (GUI code remains the same as previous)
class VoiceClipperGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ISEF Kuchipudi Dataset Builder")
        self.setMinimumWidth(700)
        self.ref_path = ""
        self.big_dir = ""
        self.out_dir = os.path.join(os.getcwd(), "dataset_output")
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        h1, h2, h3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        self.lbl_ref = QLabel("<b>1.</b> Reference File")
        btn_ref = QPushButton("Select File")
        btn_ref.clicked.connect(self.select_ref)
        h1.addWidget(self.lbl_ref, 1); h1.addWidget(btn_ref)
        self.lbl_big = QLabel("<b>2.</b> Big Files Folder")
        btn_big = QPushButton("Select Folder")
        btn_big.clicked.connect(self.select_big)
        h2.addWidget(self.lbl_big, 1); h2.addWidget(btn_big)
        self.lbl_out = QLabel(f"<b>3.</b> Output")
        btn_out = QPushButton("Change")
        btn_out.clicked.connect(self.select_out)
        h3.addWidget(self.lbl_out, 1); h3.addWidget(btn_out)
        self.btn_run = QPushButton("GENERATE DATASET")
        self.btn_run.setStyleSheet("background-color: #2980b9; color: white; height: 50px; font-weight: bold;")
        self.btn_run.clicked.connect(self.start)
        self.pbar = QProgressBar()
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #1c1c1c; color: #00ff00; font-family: 'Courier New';")
        for item in [h1, h2, h3, self.btn_run, self.pbar, self.log]:
            if isinstance(item, QHBoxLayout): layout.addLayout(item)
            else: layout.addWidget(item)
        self.setLayout(layout)

    def select_ref(self):
        self.ref_path, _ = QFileDialog.getOpenFileName(self, "Select Ref")
        if self.ref_path: self.lbl_ref.setText(f"REF: {os.path.basename(self.ref_path)}")
    def select_big(self):
        self.big_dir = QFileDialog.getExistingDirectory(self, "Select Big Files Folder")
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