import sys, os, subprocess, shutil, librosa, numpy as np, time
from pathlib import Path
import scipy.signal

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QLabel, QTextEdit, QProgressBar,
                             QHBoxLayout, QLineEdit)
from PySide6.QtCore import QThread, Signal

# --- Configuration ---
SR = 22050
HOP_LENGTH = 512
MATCH_THRESHOLD = 0.55
BUFFER_SECONDS = 0.5
CRF = 18


def extract_audio_to_wav(input_path, output_wav_path):
    """Use ffmpeg to extract audio as mono 22050 Hz WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav",
        str(output_wav_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def compute_chroma(wav_path):
    """Load audio and compute normalized CQT chroma features."""
    y, sr = librosa.load(str(wav_path), sr=SR)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1
    chroma = chroma / norms
    duration = librosa.get_duration(y=y, sr=sr)
    return chroma, duration


def chroma_cross_correlation(ref_chroma, master_chroma):
    """FFT-based normalized cross-correlation across 12 chroma bands."""
    ref_len = ref_chroma.shape[1]
    master_len = master_chroma.shape[1]
    if ref_len > master_len:
        return np.array([])

    num_positions = master_len - ref_len + 1

    # Sum correlations across all 12 chroma bands using FFT convolution
    scores = np.zeros(num_positions)
    for band in range(12):
        corr = scipy.signal.fftconvolve(
            master_chroma[band], ref_chroma[band, ::-1], mode='valid'
        )
        scores += corr[:num_positions]

    # Normalize
    ref_energy = np.sqrt(np.sum(ref_chroma ** 2))
    master_sq = np.sum(master_chroma ** 2, axis=0)
    cumsum = np.cumsum(master_sq)
    window_energy = np.sqrt(
        cumsum[ref_len - 1:ref_len - 1 + num_positions]
        - np.concatenate(([0], cumsum[:num_positions - 1]))
    )
    window_energy[window_energy == 0] = 1

    scores = scores / (ref_energy * window_energy)
    return scores


class ProcessingThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(str)

    def __init__(self, ref_file, big_folder, out_folder, name_prefix, start_number):
        super().__init__()
        self.ref_file = str(Path(ref_file).resolve())
        self.big_folder = str(Path(big_folder).resolve())
        self.out_folder = str(Path(out_folder).resolve())
        self.name_prefix = name_prefix
        self.start_number = start_number

    def run(self):
        temp_dir = Path(os.getcwd()) / f"temp_{int(time.time())}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            global_counter = self.start_number

            # 1. Extract reference audio
            self.log_signal.emit("Extracting reference audio...")
            ref_wav = temp_dir / "reference.wav"
            extract_audio_to_wav(self.ref_file, ref_wav)

            # 2. Compute reference chroma
            self.log_signal.emit("Computing reference chroma features...")
            ref_chroma, ref_duration = compute_chroma(ref_wav)
            self.log_signal.emit(f"Reference duration: {ref_duration:.2f}s | Chroma frames: {ref_chroma.shape[1]}")

            # 3. Process master video files
            extensions = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
            big_files = sorted([f for f in os.listdir(self.big_folder) if f.lower().endswith(extensions)])

            if not big_files:
                self.finished_signal.emit("No video files found in the selected folder.")
                return

            self.log_signal.emit(f"Found {len(big_files)} master video(s) to process.")

            for i, big_file in enumerate(big_files):
                big_path = str(Path(self.big_folder) / big_file)
                self.log_signal.emit(f"\nProcessing: {big_file}")

                # Extract master audio
                master_wav = temp_dir / f"master_{i}.wav"
                try:
                    extract_audio_to_wav(big_path, master_wav)
                except subprocess.CalledProcessError:
                    self.log_signal.emit(f"  Failed to extract audio from {big_file}, skipping.")
                    continue

                # Compute master chroma
                self.log_signal.emit("  Computing chroma features...")
                master_chroma, master_duration = compute_chroma(master_wav)
                self.log_signal.emit(f"  Master duration: {master_duration:.2f}s | Chroma frames: {master_chroma.shape[1]}")

                # Cross-correlation
                self.log_signal.emit("  Running cross-correlation matching...")
                scores = chroma_cross_correlation(ref_chroma, master_chroma)

                if len(scores) == 0:
                    self.log_signal.emit("  Reference is longer than master, skipping.")
                    continue

                max_score = np.max(scores)
                self.log_signal.emit(f"  Max correlation score: {max_score:.4f} (threshold: {MATCH_THRESHOLD})")

                # Find peaks
                min_distance_frames = int(ref_chroma.shape[1] * 0.8)
                peaks, properties = scipy.signal.find_peaks(
                    scores,
                    height=MATCH_THRESHOLD,
                    distance=min_distance_frames,
                    prominence=0.05
                )

                if len(peaks) == 0:
                    self.log_signal.emit("  No matches found.")
                    continue

                self.log_signal.emit(f"  Found {len(peaks)} match(es)!")

                for peak_frame in peaks:
                    start_time = peak_frame * HOP_LENGTH / SR
                    clip_duration = ref_duration + BUFFER_SECONDS
                    peak_score = scores[peak_frame]

                    output_name = f"{self.name_prefix}_{global_counter:04d}.mp4"
                    output_path = os.path.join(self.out_folder, output_name)

                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{start_time:.3f}",
                        "-i", big_path,
                        "-t", f"{clip_duration:.3f}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF),
                        "-c:a", "aac", "-avoid_negative_ts", "make_zero",
                        output_path
                    ]
                    subprocess.run(ffmpeg_cmd, capture_output=True)

                    self.log_signal.emit(
                        f"  Exported: {output_name} | Start: {start_time:.2f}s | "
                        f"Duration: {clip_duration:.2f}s | Score: {peak_score:.4f}"
                    )
                    global_counter += 1

                # Clean up master wav to save disk space
                if master_wav.exists():
                    master_wav.unlink()

                self.progress_signal.emit(int(((i + 1) / len(big_files)) * 100))

            total_clips = global_counter - 1
            self.finished_signal.emit(f"Done! Extracted {total_clips} clip(s) total.")

        except Exception as e:
            self.finished_signal.emit(f"Error: {str(e)}")
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


class SongMatcherGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kuchipudi Song Extractor")
        self.setMinimumWidth(700)
        self.ref_path = ""
        self.big_dir = ""
        self.out_dir = os.path.join(os.getcwd(), "dataset_output")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        h1, h2, h3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()

        self.lbl_ref = QLabel("<b>1.</b> Select Reference Audio/Video")
        btn_ref = QPushButton("Select File")
        btn_ref.clicked.connect(self.select_ref)
        h1.addWidget(self.lbl_ref, 1)
        h1.addWidget(btn_ref)

        self.lbl_big = QLabel("<b>2.</b> Select Master Videos Folder")
        btn_big = QPushButton("Select Folder")
        btn_big.clicked.connect(self.select_big)
        h2.addWidget(self.lbl_big, 1)
        h2.addWidget(btn_big)

        self.lbl_out = QLabel("<b>3.</b> Output Folder")
        btn_out = QPushButton("Change")
        btn_out.clicked.connect(self.select_out)
        h3.addWidget(self.lbl_out, 1)
        h3.addWidget(btn_out)

        # --- Output Naming ---
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("<b>4.</b> Output Name:"))
        self.txt_name = QLineEdit("kd_DhidhiThai_Full")
        self.txt_name.setPlaceholderText("e.g. kd_DhidhiThai_Full")
        h_name.addWidget(self.txt_name, 1)
        h_name.addWidget(QLabel("_"))
        self.txt_start_num = QLineEdit("1")
        self.txt_start_num.setFixedWidth(80)
        self.txt_start_num.setPlaceholderText("Start #")
        h_name.addWidget(self.txt_start_num)
        self.lbl_preview = QLabel("")
        h_name.addWidget(self.lbl_preview)

        self.txt_name.textChanged.connect(self._update_name_preview)
        self.txt_start_num.textChanged.connect(self._update_name_preview)
        self._update_name_preview()

        self.btn_run = QPushButton("FIND AND EXTRACT MATCHES")
        self.btn_run.setStyleSheet("background-color: #27ae60; color: white; height: 50px; font-weight: bold;")
        self.btn_run.clicked.connect(self.start)

        self.pbar = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #1c1c1c; color: #39FF14; font-family: 'Consolas';")

        for item in [h1, h2, h3, h_name, self.btn_run, self.pbar, self.log]:
            if isinstance(item, QHBoxLayout):
                layout.addLayout(item)
            else:
                layout.addWidget(item)
        self.setLayout(layout)

    def _update_name_preview(self):
        name = self.txt_name.text().strip() or "Name"
        try:
            num = int(self.txt_start_num.text().strip())
        except ValueError:
            num = 1
        self.lbl_preview.setText(f"Preview: <b>{name}_{num:04d}.mp4</b>")

    def select_ref(self):
        self.ref_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio/Video",
            filter="Media Files (*.mp4 *.mov *.mkv *.avi *.webm *.mp3 *.wav *.flac *.aac *.m4a *.ogg *.wma);;All Files (*)"
        )
        if self.ref_path:
            self.lbl_ref.setText(f"REF: {os.path.basename(self.ref_path)}")

    def select_big(self):
        self.big_dir = QFileDialog.getExistingDirectory(self, "Select Master Videos Folder")
        if self.big_dir:
            self.lbl_big.setText(f"FOLDER: {os.path.basename(self.big_dir)}")

    def select_out(self):
        self.out_dir = QFileDialog.getExistingDirectory(self, "Output Folder")
        if self.out_dir:
            self.lbl_out.setText(f"OUTPUT: {os.path.basename(self.out_dir)}")

    def start(self):
        if not self.ref_path or not self.big_dir:
            return
        name_prefix = self.txt_name.text().strip() or "output"
        try:
            start_number = int(self.txt_start_num.text().strip())
        except ValueError:
            start_number = 1
        self.btn_run.setEnabled(False)
        self.worker = ProcessingThread(self.ref_path, self.big_dir, self.out_dir, name_prefix, start_number)
        self.worker.log_signal.connect(self.log.append)
        self.worker.progress_signal.connect(self.pbar.setValue)
        self.worker.finished_signal.connect(lambda m: [self.log.append(m), self.btn_run.setEnabled(True)])
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SongMatcherGUI()
    window.show()
    sys.exit(app.exec())
