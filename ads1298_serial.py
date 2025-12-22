# ============== USER CONFIG ==============
GAIN           = 1.0
TRANSPORT      = "ble"  # "uart" or "ble"
VIS_LIB        = "pg"    # "mpl" or "pg"
PREVIEW_5COL   = True
WINDOW_CH7     = 0.1
WINDOW_CH8     = 0.1
WINDOW_PREVIEW = 0.05
SAVE_CSV       = False   # True or False
# ========================================

import sys, time, struct
from collections import deque
from threading import Thread, Event
from typing import Optional, List
import numpy as np
import csv

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

try:
    from bleak import BleakScanner, BleakClient
except Exception:
    BleakScanner = BleakClient = None

# Matplotlib
import matplotlib.pyplot as plt

# Pyqtgraph
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtGui, QtCore

# ML / Torch
import torch
import os

# Try to import model factory - robustly handle if potential path issues
try:
    from ml_training.model_factory import create_model
    from ml_training.dataset import preprocess_signal
except ImportError:
    create_model = None
    preprocess_signal = None
    print("[WARN] Could not import ml_training modules. Inference will be disabled.")

class RealTimePPGDetector:
    """
    Real-time inference engine for V4.0 Dual-Stream Model.
    Maintains a buffer and uses a shared model instance.
    """
    def __init__(self, model_instance=None, model_path=None, device='cpu', 
                 buffer_size=2560, inference_stride=1000, name="Detector"):
        self.device = torch.device(device)
        self.buffer_size = buffer_size # 2.56s @ 1000Hz
        self.stride = inference_stride
        self.buffer = deque(maxlen=buffer_size)
        self.samples_since_last_inference = 0
        self.name = name
        
        # Shared model or load new
        if model_instance:
            self.model = model_instance
        elif model_path and create_model:
            self.model = self._load_model(model_path)
        else:
            self.model = None
            
        self.pulse_labels = {0: 'Sinus (N)', 1: 'Premature (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
        self.noise_labels = {0: 'Clean', 1: 'Baseline', 2: 'Forearm', 3: 'Hand', 4: 'HighFreq'}
        
    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"[WARN] Model {path} not found.")
            return None
        try:
            print(f"[ML] Loading V4.0 Model from {path}...")
            model = create_model('dual', in_channels=34, n_classes_seg=5, n_classes_clf=5, attention=True)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"[ERR] Failed to load model: {e}")
            return None

    def process_new_data(self, new_samples):
        if self.model is None or preprocess_signal is None: return None
        
        self.buffer.extend(new_samples)
        self.samples_since_last_inference += len(new_samples)
        
        if len(self.buffer) == self.buffer_size and self.samples_since_last_inference >= self.stride:
            self.samples_since_last_inference = 0
            return self._run_inference()
        return None
    
    def _run_inference(self):
        signal_np = np.array(self.buffer, dtype=np.float32)
        
        # Preprocess (CWT + Norm)
        # preprocess_signal expects returns torch tensor [34, L]
        try:
            tensor = preprocess_signal(signal_np) 
            tensor = tensor.unsqueeze(0).to(self.device) # [1, 34, L]
            
            with torch.no_grad():
                pred_clf, pred_seg = self.model(tensor)
                
            # Classification
            prob_clf = torch.softmax(pred_clf, dim=1)
            pred_p = prob_clf.argmax(1).item()
            conf_p = prob_clf[0, pred_p].item()
            
            # Segmentation
            pred_mask = pred_seg.argmax(1).squeeze().cpu().numpy()
            
            # Analysis
            unique, counts = np.unique(pred_mask, return_counts=True)
            noise_summary = []
            has_noise = False
            total_pts = len(pred_mask)
            
            for u, c in zip(unique, counts):
                if u == 0: continue
                ratio = c / total_pts
                if ratio > 0.05: # >5% coverage
                    noise_summary.append(f"{self.noise_labels.get(u, str(u))}({ratio:.0%})")
                    has_noise = True
                    
            return {
                "name": self.name,
                "pulse_type": self.pulse_labels.get(pred_p, str(pred_p)),
                "pulse_conf": conf_p,
                "has_noise": has_noise,
                "noise_str": ", ".join(noise_summary) if noise_summary else "None"
            }
            
        except Exception as e:
            print(f"[Inference Error {self.name}] {e}")
            return None

# ... (Previous Constants kept) ...

# -------------- Data model ---------------
class DataModel:
    def __init__(self, window_s: float, expected_sps: int, samp_dt: float,
                 vref: Optional[float], gain: float):
        self.window = window_s
        self.expected_sps = max(1, expected_sps)
        self.samp_dt = samp_dt
        self.vref = vref
        self.gain = gain
        self.window_samples_ch = [
            max(int(WINDOW_CH8 * self.expected_sps * 1.25), 1000),  # CH5
            max(int(WINDOW_CH8 * self.expected_sps * 1.25), 1000),  # CH6
            max(int(WINDOW_CH7 * self.expected_sps * 1.25), 1000),  # CH7 full
            max(int(WINDOW_CH8 * self.expected_sps * 1.25), 1000),  # CH8
        ]
        self.ch_bufs = [deque(maxlen=n) for n in self.window_samples_ch]

        # Demux buckets for CH7
        self.window_samples_ch7 = max(int(WINDOW_CH7 * self.expected_sps / CYCLE_STEPS * 1.25), 200)
        self.ch7_buckets = [deque(maxlen=self.window_samples_ch7) for _ in range(CYCLE_STEPS)]
        
        self.last_rx_ts = 0.0
        self.phase_offset_frames = 0
        self.demux_fixed = False
        self.current_phase = 0xFF
        self.ch8_zero_flags = deque(maxlen=200)
        
        # Detector Setup for CH7 Slices (LED 1-4)
        self.detectors = []
        self._init_detectors()

    def _init_detectors(self):
        # Load Shared Model Once
        if create_model:
            # Assuming 'output/v4_best_model.pth' is the definitive v4 model
            # Use absolute path if possible or relative
            model_path = os.path.join(os.path.dirname(__file__), "output/v4_best_model.pth")
            if not os.path.exists(model_path):
                 # Try backup
                 model_path = "output/v4_best_model.pth"
            
            # Temporary loader to get the model instance
            tmp = RealTimePPGDetector(model_path=model_path, name="Master")
            shared_model = tmp.model
            
            if shared_model:
                # Create 4 detectors sharing the same model
                for i in range(4): # LED1-4
                    self.detectors.append(RealTimePPGDetector(model_instance=shared_model, name=f"LED{i+1}"))
                print(f"[AI] Initialized 4 Detectors for CH7 Slices (Sharing V4 Model)")
            else:
                print("[AI] Failed to load model. Inference disabled.")

    def set_detector(self, detector):
        pass # Deprecated single detector setter

    def clear(self):
        for dq in self.ch7_buckets:
            dq.clear()
        for dq in self.ch_bufs:
            dq.clear()
        # Detectors buffers also clear? better
        for d in self.detectors:
            d.buffer.clear()

    def append_frame(self, i_local: int, ch_counts: List[int]):
        # ch_counts[0..3] = CH5, CH6, CH7, CH8
        raw = [ch_counts[i] if i < len(ch_counts) else 0 for i in range(4)]
        vals = [
            counts_to_volts(raw[i], self.vref, self.gain) if self.vref is not None else float(raw[i])
            for i in range(4)
        ]
        ts = time.time()

        self.ch8_zero_flags.append(int(raw[3] == 0))

        # Phasing
        if self.demux_fixed and self.current_phase in (0,1,2,3,4):
            bucket = self.current_phase
        else:
            bucket = (((i_local + self.phase_offset_frames) - 1) // STEP_FRAMES) % CYCLE_STEPS

        # Append to bucket buffer
        self.ch7_buckets[bucket].append((i_local, vals[2], ts))

        # Append to full channel buffers
        for ch_idx in range(4):
            self.ch_bufs[ch_idx].append((i_local, vals[ch_idx], ts))

        self.last_rx_ts = ts
        
        # Inference Logic: Feed CH7 slices
        # bucket 0..3 are LEDs. bucket 4 is Dark (usually).
        if bucket < 4 and self.detectors:
            # Feed current val to the correct detector
            # Need to downsample? Preprocess does resampling? 
            # Original code "Downsample 2000 -> 1000 Hz" via "i_local % 2 == 0" check.
            # Assuming 2000Hz native.
            # We want 1000Hz for model.
            # Only feed every 2nd sample
            if i_local % 2 == 0:
                res = self.detectors[bucket].process_new_data([vals[2]])
                if res:
                    print(f"\n[AI] {res['name']}: {res['pulse_type']} ({res['pulse_conf']:.1%}) | Noise: {res['noise_str']}")

    
    def is_ch7only(self) -> bool:
        if not self.ch8_zero_flags: return False
        ratio = sum(self.ch8_zero_flags) / len(self.ch8_zero_flags)
        return ratio > 0.98

    def get_window_bucket(self, b: int):
        data = list(self.ch7_buckets[b])
        if not data: return np.array([]), np.array([]), np.array([]), np.array([])
        idxs = np.array([i for (i, *_rest) in data], dtype=np.int64)
        ys   = np.array([y for (_, y, _ts) in data], dtype=np.float32)
        ts   = np.array([ts for (*_, ts) in data], dtype=np.float64)
        xs   = (idxs - idxs[0]) * self.samp_dt
        return xs, ys, ts, idxs

    def get_window_ch(self, ch_idx: int):
        if ch_idx < 0 or ch_idx >= len(self.ch_bufs):
            return np.array([]), np.array([]), np.array([]), np.array([])
        data = list(self.ch_bufs[ch_idx])
        if not data:
            return np.array([]), np.array([]), np.array([]), np.array([])
        idxs = np.array([i for (i, *_rest) in data], dtype=np.int64)
        ys   = np.array([y for (_, y, _ts) in data], dtype=np.float32)
        ts   = np.array([ts for (*_, ts) in data], dtype=np.float64)
        xs   = (idxs - idxs[0]) * self.samp_dt
        return xs, ys, ts, idxs

# -------------- Transport readers ---------------
def make_decoder(model: DataModel):
    # ---- CSV 相关状态（闭包变量）----
    csv_file = None
    csv_writer = None

    def open_new_csv():
        nonlocal csv_file, csv_writer
        if not SAVE_CSV:
            return  # 开关关闭时，直接什么都不做

        # 如果之前有文件，先关掉
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

        ts_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ads_{ts_str}.csv"

        csv_file = open(filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_index",
                            "ch5_counts", "ch6_counts", "ch7_counts", "ch8_counts",
                            "host_time"])
        print(f"[CSV] 新建文件: {filename}")

    def on_sync(version: int, channels: int):
        model.clear()
        print(f"[SYNC] version={version}, channels={channels}, demux=({CYCLE_STEPS} steps @ {STEP_FRAMES} frames/step)")
        # 每次开始采数时，新建一个按时间命名的 CSV（只有在 SAVE_CSV=True 时才会真正创建）
        open_new_csv()

    def on_time(uptime_ms: int, frame_count: int, i_local: int, last_local: int, last_remote: int):
        print(f"[TIME] uptime={uptime_ms} ms, remote_frames={frame_count}, local_frames={i_local}")
        model.demux_fixed = model.is_ch7only()

    def on_frame(i_local: int, ch_counts: List[int]):
        nonlocal csv_writer
        # 先喂给模型（用来画图）
        model.append_frame(i_local, ch_counts)

        # 再写 CSV（如果开启的话）
        if SAVE_CSV and csv_writer is not None:
            c = [ch_counts[i] if i < len(ch_counts) else 0 for i in range(4)]
            csv_writer.writerow([i_local, c[0], c[1], c[2], c[3], time.time()])

    def on_phase_update(offset_frames: int):
        model.phase_offset_frames = offset_frames

    def on_phase_value(phase: int):
        model.current_phase = phase   # 0..4 or 0xFF

    return StreamDecoder(on_frame=on_frame,
                         on_sync=on_sync,
                         on_time=on_time,
                         on_phase_update=on_phase_update,
                         on_phase_value=on_phase_value)

def start_uart_reader(port: str, baud: int, stop_evt: Event, decoder: StreamDecoder):
    def run():
        if serial is None:
            print("[ERR] pyserial not installed.", file=sys.stderr)
            stop_evt.set(); return
        p = port
        if isinstance(p, str) and p.lower() == "auto":
            ap = autodetect_port()
            if not ap:
                print("[ERR] auto-detect failed; please set PORT.", file=sys.stderr); return
            p = ap
        try:
            ser = serial.Serial(p, baudrate=baud, timeout=1, rtscts=False, dsrdtr=False,
                                exclusive=True if sys.platform.startswith("linux") else False)
            ser.reset_input_buffer()
            print(f"[info] opened {p} @ {baud} baud")
        except Exception as e:
            print(f"[ERR] Could not open serial port {p}: {e}", file=sys.stderr)
            stop_evt.set(); return
        try:
            while not stop_evt.is_set():
                chunk = ser.read(4096)
                if chunk:
                    decoder.feed(chunk)
        finally:
            try: ser.close()
            except Exception: pass
    t = Thread(target=run, daemon=True)
    t.start()
    return t

def start_ble_reader(stop_evt: Event, decoder: StreamDecoder):
    if BleakScanner is None or BleakClient is None:
        print("[ERR] bleak not installed.", file=sys.stderr)
        stop_evt.set()
        t = Thread(target=lambda: None, daemon=True); t.start(); return t

    def thread_target():
        import asyncio
        async def run():
            def match(dev, adv):
                if BLE_NAME and dev.name == BLE_NAME: return True
                return (UART_SERVICE_UUID.lower() in (adv.service_uuids or []))
            device = await BleakScanner.find_device_by_filter(match)
            if not device:
                print("[ERR] BLE device not found"); stop_evt.set(); return
            def on_notify(_, data: bytearray):
                decoder.feed(data)
            async with BleakClient(device) as cli:
                await cli.start_notify(UART_TX_CHAR_UUID, on_notify)
                print(f"[info] BLE connected: {device.name or device.address}")
                try:
                    while not stop_evt.is_set():
                        await asyncio.sleep(0.05)
                finally:
                    try: await cli.stop_notify(UART_TX_CHAR_UUID)
                    except Exception: pass
        asyncio.run(run())
    t = Thread(target=thread_target, daemon=True)
    t.start()
    return t

# -------------- Matplotlib ---------------
class MplFrontend:
    def __init__(self, model: DataModel, preview5: bool = True, title: str = "Viewer"):
        self.model = model
        self.preview5 = preview5
        self.led_sel = 0      # 当前放大的 CH7 bucket（0..4 -> LED1..4 + DARK）
        self.paused = False

        self.fig = plt.figure(figsize=(14, 6))

        # ------- 外层 3×1：上=CH5/6，中=CH7/8，下=预览 -------
        gs_outer = self.fig.add_gridspec(nrows=3, ncols=1,
                                         height_ratios=[1, 1, 1 if preview5 else 0])

        # 顶部：CH5 & CH6 等宽
        gs_top = gs_outer[0].subgridspec(1, 2)
        self.ax5 = self.fig.add_subplot(gs_top[0, 0])
        self.ax6 = self.fig.add_subplot(gs_top[0, 1])

        # 中部：CH7 (bucket) & CH8 等宽
        gs_mid = gs_outer[1].subgridspec(1, 2)
        self.ax7 = self.fig.add_subplot(gs_mid[0, 0])
        self.ax8 = self.fig.add_subplot(gs_mid[0, 1])

        # 底部：4 个预览（LED1-4），不再画 DARK
        self.prev_axes = []
        self.prev_lines = []
        if preview5:
            gs_bot = gs_outer[2].subgridspec(1, 4)
            for c in range(4):                # 只做 4 列
                ax = self.fig.add_subplot(gs_bot[0, c])
                ln, = ax.plot([], [], lw=1.0, color='0.2')
                ax.set_title(STEP_LABELS[c], fontsize=18)   # STEP_LABELS[0..3] = LED1..4
                ax.tick_params(labelsize=8)
                self.prev_axes.append(ax)
                self.prev_lines.append(ln)

        # ------- 四个大通道的曲线（CH5/6 也蓝色） -------
        # CH5 / CH6 / CH8 蓝色，CH7 红色
        self.line5, = self.ax5.plot([], [], lw=1.5, color='b')
        self.ax5.set_ylabel("CH5 Voltage (V)" if VREF is not None else "CH5 Counts")
        self.ax5.set_title("CH5")

        self.line6, = self.ax6.plot([], [], lw=1.5, color='b')
        self.ax6.set_ylabel("CH6 Voltage (V)" if VREF is not None else "CH6 Counts")
        self.ax6.set_title("CH6")

        self.line7, = self.ax7.plot([], [], lw=2.0, color='r')
        self.ax7.set_ylabel("CH7 Voltage (V)" if VREF is not None else "CH7 Counts")
        base_sz = self.ax7.title.get_fontsize() or 12
        self._ch7_title_size = base_sz * 2
        self.ax7.set_title(f"CH7 ({STEP_LABELS[self.led_sel]})",
                           fontsize=self._ch7_title_size)

        self.line8, = self.ax8.plot([], [], lw=2.0, color='b')
        self.ax8.set_ylabel("CH8 Voltage (V)" if VREF is not None else "CH8 Counts")
        self.ax8.set_xlabel("Time (s)")
        self.ax8.set_title("CH8")

        # ------- Y 轴范围 -------
        if VREF is not None:
            vmax = (VREF / max(GAIN, 1e-12))
            for ax in [self.ax5, self.ax6, self.ax7, self.ax8]:
                ax.set_ylim(-vmax * 1.05, +vmax * 1.05)
            if preview5:
                for ax in self.prev_axes:
                    ax.set_ylim(-vmax * 1.05, +vmax * 1.05)
        else:
            for ax in [self.ax5, self.ax6, self.ax7, self.ax8]:
                ax.autoscale(enable=False, axis="y")
            if preview5:
                for ax in self.prev_axes:
                    ax.autoscale(enable=False, axis="y")

        # 坐标轴刻度字体
        axes_all = [self.ax5, self.ax6, self.ax7, self.ax8] + (self.prev_axes if self.preview5 else [])
        for ax in axes_all:
            ax.tick_params(axis='both', labelsize=12)

        # 键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.suptitle(
            f"{TITLE} [MPL] | CH7={STEP_LABELS[self.led_sel]} | preview={'ON' if preview5 else 'OFF'}"
        )
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ---------- 键盘控制 ----------
    def on_key(self, event):
        if event.key in ('1', '2', '3', '4', '5'):
            self.led_sel = int(event.key) - 1
            self.fig.suptitle(
                f"{TITLE} [MPL] | CH7={STEP_LABELS[self.led_sel]} | preview={'ON' if self.preview5 else 'OFF'}"
            )
            self.ax7.set_title(f"CH7 ({STEP_LABELS[self.led_sel]})",
                               fontsize=self._ch7_title_size)
            self.fig.canvas.draw_idle()
        elif event.key == 'p':
            self.paused = not self.paused
            print(f"[info] paused={self.paused}")
        elif event.key == 's':
            fname = time.strftime("plot_%Y%m%d_%H%M%S.png")
            self.fig.savefig(fname, dpi=150)
            print(f"[info] saved snapshot: {fname}")
        elif event.key == 'v':
            self.preview5 = not self.preview5
            print(f"[info] preview5={self.preview5} (restart script to fully re-layout)")

    # ---------- 预览 decimate ----------
    def _decimate(self, xs: np.ndarray, ys: np.ndarray, maxpts: int):
        n = xs.size
        if n <= maxpts:
            return xs, ys
        step = max(1, n // maxpts)
        return xs[::step], ys[::step]

    # ---------- 主循环 ----------
    def loop(self):
        last_log = time.time()
        while True:
            if self.paused:
                plt.pause(0.05)
                continue

            now = time.time()

            # --- Idle：超时没数据，画零线 ---
            if (now - self.model.last_rx_ts) >= IDLE_TIMEOUT:
                N = max(int(WINDOW_CH8 * self.model.expected_sps), 400)

                xs5z = np.linspace(0.0, WINDOW_CH8, N, dtype=float)
                z = np.zeros(N, dtype=float)
                self.line5.set_data(xs5z, z)
                self.ax5.set_xlim(0.0, WINDOW_CH8)

                xs6z = np.linspace(0.0, WINDOW_CH8, N, dtype=float)
                self.line6.set_data(xs6z, z)
                self.ax6.set_xlim(0.0, WINDOW_CH8)

                N7 = max(int(WINDOW_CH7 * self.model.expected_sps / CYCLE_STEPS), 400)
                xs7z = np.linspace(0.0, WINDOW_CH7, N7, dtype=float)
                z7 = np.zeros(N7, dtype=float)
                self.line7.set_data(xs7z, z7)
                self.ax7.set_xlim(0.0, WINDOW_CH7)

                xs8z = np.linspace(0.0, WINDOW_CH8, N, dtype=float)
                self.line8.set_data(xs8z, z)
                self.ax8.set_xlim(0.0, WINDOW_CH8)

                if self.preview5:
                    Np = max(int(WINDOW_PREVIEW * self.model.expected_sps / CYCLE_STEPS), 200)
                    xsp = np.linspace(0.0, WINDOW_PREVIEW, Np, dtype=float)
                    zp = np.zeros(Np, dtype=float)
                    for ax, ln in zip(self.prev_axes, self.prev_lines):
                        ln.set_data(xsp, zp)
                        ax.set_xlim(0.0, WINDOW_PREVIEW)

                plt.pause(0.05)
                continue

            # --- 正常刷新：取数据 ---
            xs5, ys5, ts5, idxs5 = self.model.get_window_ch(0)  # CH5
            xs6, ys6, ts6, idxs6 = self.model.get_window_ch(1)  # CH6
            xs7, ys7, ts7, idxs7 = self.model.get_window_bucket(self.led_sel)  # CH7 bucket
            xs8, ys8, ts8, idxs8 = self.model.get_window_ch(3)  # CH8

            # CH5
            if xs5.size:
                if BREAK_ON_GAP and idxs5.size >= 2:
                    mask = np.diff(idxs5) > 1
                    if mask.any():
                        insert_at = np.where(mask)[0] + 1
                        mid_x = xs5[insert_at - 1] + (xs5[insert_at] - xs5[insert_at - 1]) * 0.5
                        xs5 = np.insert(xs5, insert_at, mid_x)
                        ys5 = np.insert(ys5, insert_at, np.nan)
                self.line5.set_data(xs5, ys5)
                xmax5 = xs5[-1]
                self.ax5.set_xlim(max(0.0, xmax5 - WINDOW_CH8), xmax5)

            # CH6
            if xs6.size:
                if BREAK_ON_GAP and idxs6.size >= 2:
                    mask = np.diff(idxs6) > 1
                    if mask.any():
                        insert_at = np.where(mask)[0] + 1
                        mid_x = xs6[insert_at - 1] + (xs6[insert_at] - xs6[insert_at - 1]) * 0.5
                        xs6 = np.insert(xs6, insert_at, mid_x)
                        ys6 = np.insert(ys6, insert_at, np.nan)
                self.line6.set_data(xs6, ys6)
                xmax6 = xs6[-1]
                self.ax6.set_xlim(max(0.0, xmax6 - WINDOW_CH8), xmax6)

            # CH7 bucket
            if xs7.size:
                if BREAK_ON_GAP and idxs7.size >= 2:
                    mask = np.diff(idxs7) > 1
                    if mask.any():
                        insert_at = np.where(mask)[0] + 1
                        mid_x = xs7[insert_at - 1] + (xs7[insert_at] - xs7[insert_at - 1]) * 0.5
                        xs7 = np.insert(xs7, insert_at, mid_x)
                        ys7 = np.insert(ys7, insert_at, np.nan)
                self.line7.set_data(xs7, ys7)
                xmax7 = xs7[-1]
                self.ax7.set_xlim(max(0.0, xmax7 - WINDOW_CH7), xmax7)

            # CH8 full
            if xs8.size:
                if BREAK_ON_GAP and idxs8.size >= 2:
                    mask = np.diff(idxs8) > 1
                    if mask.any():
                        insert_at = np.where(mask)[0] + 1
                        mid_x = xs8[insert_at - 1] + (xs8[insert_at] - xs8[insert_at - 1]) * 0.5
                        xs8 = np.insert(xs8, insert_at, mid_x)
                        ys8 = np.insert(ys8, insert_at, np.nan)
                self.line8.set_data(xs8, ys8)
                xmax8 = xs8[-1]
                self.ax8.set_xlim(max(0.0, xmax8 - WINDOW_CH8), xmax8)

            # 预览：只画 LED1-4 对应的前 4 个 bucket
            if self.preview5:
                for b in range(len(self.prev_axes)):    # 现在是 4
                    xsb, ysb, tsb, idxsb = self.model.get_window_bucket(b)
                    if xsb.size:
                        if BREAK_ON_GAP and idxsb.size >= 2:
                            mask = np.diff(idxsb) > 1
                            if mask.any():
                                insert_at = np.where(mask)[0] + 1
                                mid_x = xsb[insert_at - 1] + (xsb[insert_at] - xsb[insert_at - 1]) * 0.5
                                xsb = np.insert(xsb, insert_at, mid_x)
                                ysb = np.insert(ysb, insert_at, np.nan)
                        xsb_d, ysb_d = self._decimate(xsb, ysb, PREVIEW_MAXPTS)
                        self.prev_lines[b].set_data(xsb_d, ysb_d)
                        xmaxp = xsb[-1]
                        self.prev_axes[b].set_xlim(max(0.0, xmaxp - WINDOW_PREVIEW), xmaxp)

            # 打印窗口内估算 SPS（基于 CH8）
            if time.time() - last_log >= 1.0 and xs8.size > 1:
                dt = xs8[-1] - xs8[0]
                sps = (xs8.size / dt) if dt > 1e-6 else 0
                print(f"[info][MPL] window ~{sps:.0f} fps, CH7={STEP_LABELS[self.led_sel]}")
                last_log = time.time()

            plt.pause(0.01)

# -------------- PyQtGraph ---------------
class PgFrontend(QtWidgets.QMainWindow):
    def __init__(self, model: DataModel, preview5: bool = True, title: str = "Viewer"):
        super().__init__()
        self.model = model
        self.preview5 = preview5
        self.led_sel = 0
        self.paused = False

        self.setWindowTitle(f"{TITLE} [PG]")
        self.win = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.win)

        # ---------- 四个大通道 ----------
        # 第一行：CH5, CH6 等宽
        self.p5 = self.win.addPlot(row=0, col=0, colspan=2, title="CH5")
        self.p5.showGrid(x=True, y=True)
        self.p5.setClipToView(True)
        self.p5.setDownsampling(mode='peak')
        self.curve5 = self.p5.plot(pen=pg.mkPen('b', width=2))   # 蓝

        self.p6 = self.win.addPlot(row=0, col=2, colspan=2, title="CH6")
        self.p6.showGrid(x=True, y=True)
        self.p6.setClipToView(True)
        self.p6.setDownsampling(mode='peak')
        self.curve6 = self.p6.plot(pen=pg.mkPen('b', width=2))   # 蓝

        # 第二行：CH7 (bucket), CH8 等宽
        self.p7 = self.win.addPlot(row=1, col=0, colspan=2, title="CH7")
        self.p7.showGrid(x=True, y=True)
        self.p7.setClipToView(True)
        self.p7.setDownsampling(mode='peak')
        self.curve7 = self.p7.plot(pen=pg.mkPen('r', width=3))   # 红

        self.p8 = self.win.addPlot(row=1, col=2, colspan=2, title="CH8")
        self.p8.showGrid(x=True, y=True)
        self.p8.setClipToView(True)
        self.p8.setDownsampling(mode='peak')
        self.curve8 = self.p8.plot(pen=pg.mkPen('b', width=3))   # 蓝

        # ---------- 预览 4 个 bucket：LED1..LED4 ----------
        self.prev_plots = []
        self.prev_curves = []
        if preview5:
            for c in range(4):   # 只做 4 列
                p = self.win.addPlot(row=2, col=c, title=STEP_LABELS[c])
                p.setTitle(STEP_LABELS[c], size="18pt")
                p.showGrid(x=True, y=True)
                p.setClipToView(True)
                p.setDownsampling(mode='peak')
                self.prev_plots.append(p)
                self.prev_curves.append(p.plot(pen=pg.mkPen('#555555', width=2)))

        # Y 范围
        if VREF is not None:
            vmax = (VREF / max(GAIN, 1e-12))
            for p in [self.p5, self.p6, self.p7, self.p8] + self.prev_plots:
                p.setYRange(-vmax * 1.05, +vmax * 1.05)

        # 快捷键
        for i in range(5):   # 仍然允许按 1..5 选择包括 DARK 的 bucket
            QtGui.QShortcut(QtGui.QKeySequence(str(i + 1)), self,
                            activated=lambda i=i: self.select_bucket(i))
        QtGui.QShortcut(QtGui.QKeySequence('P'), self, activated=self.toggle_pause)
        QtGui.QShortcut(QtGui.QKeySequence('S'), self, activated=self.snapshot)
        QtGui.QShortcut(QtGui.QKeySequence('V'), self, activated=self.toggle_preview)

        # 刷新计时器
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(30)

        # 坐标轴字体
        tick_font = QtGui.QFont()
        tick_font.setPointSize(14)
        plots_all = [self.p5, self.p6, self.p7, self.p8] + (self.prev_plots if self.preview5 else [])
        for p in plots_all:
            p.getAxis('bottom').setTickFont(tick_font)
            p.getAxis('left').setTickFont(tick_font)

        self.update_titles()

    # ---------- 标题 ----------
    def update_titles(self):
        self.p5.setTitle("CH5", size="20pt")
        self.p6.setTitle("CH6", size="20pt")
        self.p7.setTitle(f"CH7 ({STEP_LABELS[self.led_sel]})", size="20pt")
        self.p8.setTitle("CH8", size="20pt")

    # ---------- 快捷键回调 ----------
    def select_bucket(self, i):
        self.led_sel = i
        self.update_titles()
        print(f"[info][PG] CH7 bucket -> {STEP_LABELS[i]}")

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"[info][PG] paused={self.paused}")

    def snapshot(self):
        import pyqtgraph.exporters
        exporter = pyqtgraph.exporters.ImageExporter(self.win.scene())
        exporter.parameters()['width'] = 1600
        fname = time.strftime("pg_%Y%m%d_%H%M%S.png")
        exporter.export(fname)
        print(f"[info][PG] saved snapshot: {fname}")

    def toggle_preview(self):
        self.preview5 = not self.preview5
        print("[info][PG] Toggling preview at runtime not fully supported (layout fixed at start).")

    # ---------- 刷新 ----------
    def refresh(self):
        if self.paused:
            return

        now = time.time()

        # Idle：画零线
        if (now - self.model.last_rx_ts) >= IDLE_TIMEOUT:
            xs5 = np.linspace(0.0, WINDOW_CH8, max(int(WINDOW_CH8 * self.model.expected_sps), 400), dtype=float)
            xs6 = np.linspace(0.0, WINDOW_CH8, max(int(WINDOW_CH8 * self.model.expected_sps), 400), dtype=float)
            xs7 = np.linspace(0.0, WINDOW_CH7, max(int(WINDOW_CH7 * self.model.expected_sps / CYCLE_STEPS), 400), dtype=float)
            xs8 = np.linspace(0.0, WINDOW_CH8, max(int(WINDOW_CH8 * self.model.expected_sps), 400), dtype=float)
            z5 = np.zeros_like(xs5)
            z6 = np.zeros_like(xs6)
            z7 = np.zeros_like(xs7)
            z8 = np.zeros_like(xs8)

            self.curve5.setData(xs5, z5)
            self.curve6.setData(xs6, z6)
            self.curve7.setData(xs7, z7)
            self.curve8.setData(xs8, z8)

            self.p5.setXRange(0.0, WINDOW_CH8, padding=0)
            self.p6.setXRange(0.0, WINDOW_CH8, padding=0)
            self.p7.setXRange(0.0, WINDOW_CH7, padding=0)
            self.p8.setXRange(0.0, WINDOW_CH8, padding=0)

            if self.preview5:
                xsp = np.linspace(0.0, WINDOW_PREVIEW,
                                  max(int(WINDOW_PREVIEW * self.model.expected_sps / CYCLE_STEPS), 200),
                                  dtype=float)
                zp = np.zeros_like(xsp)
                for b in range(len(self.prev_curves)):   # 只 4 个
                    self.prev_curves[b].setData(xsp, zp)
                    self.prev_plots[b].setXRange(0.0, WINDOW_PREVIEW, padding=0)
            return

        # 正常刷新
        xs5, ys5, ts5, idxs5 = self.model.get_window_ch(0)  # CH5
        xs6, ys6, ts6, idxs6 = self.model.get_window_ch(1)  # CH6
        xs7, ys7, ts7, idxs7 = self.model.get_window_bucket(self.led_sel)  # CH7 bucket
        xs8, ys8, ts8, idxs8 = self.model.get_window_ch(3)  # CH8

        if xs5.size:
            self.curve5.setData(xs5, ys5, connect='finite')
            xmax5 = xs5[-1]
            self.p5.setXRange(max(0.0, xmax5 - WINDOW_CH8), xmax5, padding=0)

        if xs6.size:
            self.curve6.setData(xs6, ys6, connect='finite')
            xmax6 = xs6[-1]
            self.p6.setXRange(max(0.0, xmax6 - WINDOW_CH8), xmax6, padding=0)

        if xs7.size:
            self.curve7.setData(xs7, ys7, connect='finite')
            xmax7 = xs7[-1]
            self.p7.setXRange(max(0.0, xmax7 - WINDOW_CH7), xmax7, padding=0)

        if xs8.size:
            self.curve8.setData(xs8, ys8, connect='finite')
            xmax8 = xs8[-1]
            self.p8.setXRange(max(0.0, xmax8 - WINDOW_CH8), xmax8, padding=0)

        if self.preview5:
            for b in range(len(self.prev_plots)):   # 只遍历 4 个 LED
                xsb, ysb, tsb, idxsb = self.model.get_window_bucket(b)
                if xsb.size:
                    self.prev_curves[b].setData(xsb, ysb, connect='finite')
                    xmaxp = xsb[-1]
                    self.prev_plots[b].setXRange(max(0.0, xmaxp - WINDOW_PREVIEW), xmaxp, padding=0)

# -------------- Main ---------------
def main():
    model = DataModel(WINDOW, EXPECTED_SPS, SAMP_DT, VREF, GAIN)
    
    # Initialize V2.0 Detector
    # User should download 'checkpoints_v2/best_model_v2.pth' from remote as 'local_best_model_v2.pth'
    detector = RealTimePPGDetector(
        model_path="local_best_model_v2.pth",
        device='cpu'  # Use CPU for real-time inference
    )
    model.set_detector(detector)
    
    decoder = make_decoder(model)
    stop_evt = Event()

    # Start transport
    if TRANSPORT.lower() == "uart":
        start_uart_reader(PORT, BAUD, stop_evt, decoder)
    else:
        start_ble_reader(stop_evt, decoder)

    if VIS_LIB.lower() == "pg":
        # PyQtGraph
        app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(
            antialias=True,
            useOpenGL=False,
            background='w',
            foreground='k'
        )
        viewer = PgFrontend(model, preview5=PREVIEW_5COL, title=TITLE)
        viewer.resize(1400, 800)
        viewer.show()
        sys.exit(app.exec())
    else:
        # Matplotlib
        frontend = MplFrontend(model, preview5=PREVIEW_5COL, title=TITLE)
        frontend.loop()

if __name__ == "__main__":
    main()
