import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2 as cv
from PIL import Image, ImageTk

from clearboard_app import ClearboardApp
from config import DEFAULT_KEYBOARD_LAYOUT_NAME, KEYBOARD_LAYOUTS


class ClearboardDesktopApp:
    video_max_width = 960
    video_max_height = 540
    frame_delay_ms = 33
    capture_delay_seconds = 1 / 30

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clearboard")
        self.root.minsize(1180, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.app = ClearboardApp()
        self.running = False
        self.current_photo = None
        self.latest_frame = None
        self.last_displayed_frame_id = 0
        self.frame_id = 0
        self.frame_error = False
        self.last_text = None
        self.display_scale = 1.0
        self.display_width = 0
        self.display_height = 0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.app_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.stop_event = threading.Event()

        self.status_var = tk.StringVar(value="Stopped")
        self.calibration_var = tk.StringVar(value="Select top-left")
        self.model_var = tk.StringVar(value="Keyboard model pending")
        self.layout_var = tk.StringVar(value=DEFAULT_KEYBOARD_LAYOUT_NAME)

        self.tuning_vars = {
            "threshold": tk.StringVar(value=str(self.app.threshold)),
            "min_press_travel": tk.StringVar(value=str(self.app.min_press_travel)),
            "press_cooldown_ms": tk.StringVar(value=str(self.app.press_cooldown_ms)),
            "smoothing_alpha": tk.StringVar(value=str(self.app.smoothing_alpha)),
            "keyboard_point_smoothing_alpha": tk.StringVar(
                value=str(self.app.keyboard_point_smoothing_alpha)
            ),
        }

        self.build_ui()
        self.refresh_status()

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, columnspan=2, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Button-1>", self.handle_video_click)

        controls = ttk.Frame(main, width=300)
        controls.grid(row=0, column=1, sticky="ns")
        controls.columnconfigure(0, weight=1)

        self.build_run_controls(controls)
        self.build_calibration_controls(controls)
        self.build_tuning_controls(controls)
        self.build_text_controls(controls)

    def build_run_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Run")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        self.stop_button = ttk.Button(frame, text="Stop", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6, pady=6)

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2)
        )
        ttk.Label(frame, textvariable=self.model_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6)
        )

    def build_calibration_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Calibration")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        layout_menu = ttk.Combobox(
            frame,
            textvariable=self.layout_var,
            values=list(KEYBOARD_LAYOUTS.keys()),
            state="readonly",
        )
        layout_menu.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        layout_menu.bind("<<ComboboxSelected>>", self.change_layout)

        ttk.Label(frame, textvariable=self.calibration_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6)
        )

        ttk.Button(frame, text="Detect Once", command=self.detect_keyboard_once).grid(
            row=2, column=0, sticky="ew", padx=6, pady=6
        )
        ttk.Button(frame, text="Lock Detected", command=self.lock_detected).grid(
            row=2, column=1, sticky="ew", padx=6, pady=6
        )
        ttk.Button(frame, text="Reset", command=self.reset_calibration).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6)
        )

    def build_tuning_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Tuning")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        fields = (
            ("Threshold", "threshold"),
            ("Min travel", "min_press_travel"),
            ("Cooldown ms", "press_cooldown_ms"),
            ("Depth smoothing", "smoothing_alpha"),
            ("Key smoothing", "keyboard_point_smoothing_alpha"),
        )

        for row, (label, key) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(frame, textvariable=self.tuning_vars[key], width=12).grid(
                row=row, column=1, sticky="ew", padx=6, pady=3
            )

        ttk.Button(frame, text="Apply", command=self.apply_tuning).grid(
            row=len(fields), column=0, columnspan=2, sticky="ew", padx=6, pady=6
        )

    def build_text_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Typed Text")
        frame.grid(row=3, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        self.text_box = tk.Text(frame, width=36, height=12, wrap="word", state="disabled")
        self.text_box.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        button_row = ttk.Frame(frame)
        button_row.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        ttk.Button(button_row, text="Correct", command=self.show_corrected_text).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(button_row, text="Clear", command=self.clear_text).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )

    def start(self):
        if self.running:
            return

        try:
            if not self.app.start(open_windows=False):
                messagebox.showerror("Clearboard", "Camera could not start.")
                self.app.cleanup()
                return
        except Exception as error:
            self.app.cleanup()
            messagebox.showerror("Clearboard", str(error))
            return

        self.running = True
        self.stop_event.clear()
        self.frame_error = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.refresh_status()
        self.update_ui()

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1)
            self.capture_thread = None

        self.app.cleanup()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_status()

    def capture_loop(self):
        while not self.stop_event.is_set():
            with self.app_lock:
                frame = self.app.process_next_frame()

            if frame is None:
                with self.frame_lock:
                    self.frame_error = True
                return

            with self.frame_lock:
                self.latest_frame = frame
                self.frame_id += 1

            time.sleep(self.capture_delay_seconds)

    def handle_missing_frame(self):
        if not self.running:
            return

        self.stop()
        messagebox.showerror("Clearboard", "Camera frame was not found.")

    def update_ui(self):
        if not self.running:
            return

        with self.frame_lock:
            frame = self.latest_frame
            frame_id = self.frame_id
            frame_error = self.frame_error

        if frame_error:
            self.handle_missing_frame()
            return

        if frame is not None and frame_id != self.last_displayed_frame_id:
            self.show_frame(frame)
            self.last_displayed_frame_id = frame_id

        self.refresh_status()
        self.root.after(self.frame_delay_ms, self.update_ui)

    def show_frame(self, frame):
        display_frame = self.fit_frame(frame)
        self.update_display_offsets(display_frame)
        rgb_frame = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self.current_photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.current_photo)

    def fit_frame(self, frame):
        height, width = frame.shape[:2]
        self.display_scale = min(
            self.video_max_width / width,
            self.video_max_height / height,
            1.0,
        )

        if self.display_scale == 1.0:
            self.display_width = width
            self.display_height = height
            return frame

        display_width = int(width * self.display_scale)
        display_height = int(height * self.display_scale)
        self.display_width = display_width
        self.display_height = display_height
        return cv.resize(frame, (display_width, display_height), interpolation=cv.INTER_AREA)

    def update_display_offsets(self, display_frame):
        display_height, display_width = display_frame.shape[:2]
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        self.display_width = display_width
        self.display_height = display_height
        self.display_offset_x = max((label_width - display_width) // 2, 0)
        self.display_offset_y = max((label_height - display_height) // 2, 0)

    def handle_video_click(self, event):
        if not self.running or self.display_scale <= 0:
            return

        display_x = event.x - self.display_offset_x
        display_y = event.y - self.display_offset_y
        if (
            display_x < 0
            or display_y < 0
            or display_x >= self.display_width
            or display_y >= self.display_height
        ):
            return

        image_x = int(display_x / self.display_scale)
        image_y = int(display_y / self.display_scale)
        with self.app_lock:
            self.app.select_manual_corner(image_x, image_y)
        self.refresh_status()

    def change_layout(self, _event=None):
        layout = KEYBOARD_LAYOUTS[self.layout_var.get()]
        with self.app_lock:
            self.app.set_keyboard_layout(layout)
        self.refresh_status()

    def detect_keyboard_once(self):
        if not self.running:
            return

        with self.app_lock:
            detected = self.app.detect_keyboard_once()

        if not detected:
            messagebox.showinfo("Clearboard", "No keyboard corners were detected.")
        self.refresh_status()

    def lock_detected(self):
        with self.app_lock:
            locked = self.app.lock_keyboard()

        if locked:
            self.refresh_status()
            return

        messagebox.showinfo("Clearboard", "No valid keyboard corners are available.")
        self.refresh_status()

    def reset_calibration(self):
        with self.app_lock:
            self.app.reset_keyboard_calibration()
        self.refresh_status()

    def apply_tuning(self):
        try:
            threshold = float(self.tuning_vars["threshold"].get())
            min_press_travel = float(self.tuning_vars["min_press_travel"].get())
            press_cooldown_ms = int(float(self.tuning_vars["press_cooldown_ms"].get()))
            smoothing_alpha = float(self.tuning_vars["smoothing_alpha"].get())
            keyboard_point_smoothing_alpha = float(
                self.tuning_vars["keyboard_point_smoothing_alpha"].get()
            )
        except ValueError:
            messagebox.showerror("Clearboard", "Tuning values must be numeric.")
            return

        if not 0 <= smoothing_alpha <= 1 or not 0 <= keyboard_point_smoothing_alpha <= 1:
            messagebox.showerror("Clearboard", "Smoothing values must be between 0 and 1.")
            return

        with self.app_lock:
            self.app.threshold = threshold
            self.app.min_press_travel = min_press_travel
            self.app.press_cooldown_ms = press_cooldown_ms
            self.app.smoothing_alpha = smoothing_alpha
            self.app.keyboard_point_smoothing_alpha = keyboard_point_smoothing_alpha

            tracker = self.app.hand_tracker
            tracker.threshold = threshold
            tracker.min_press_travel = min_press_travel
            tracker.press_cooldown_ms = press_cooldown_ms
            tracker.smoothing_alpha = smoothing_alpha
            tracker.keyboard_point_smoothing_alpha = keyboard_point_smoothing_alpha

    def show_corrected_text(self):
        with self.app_lock:
            corrected_text = self.app.correct_text(self.app.hand_tracker.text)
            self.app.hand_tracker.text = corrected_text
        self.set_text_box(corrected_text)

    def clear_text(self):
        with self.app_lock:
            self.app.clear_text()
        self.refresh_status()

    def refresh_status(self):
        if self.running:
            self.status_var.set("Running")
        else:
            self.status_var.set("Stopped")

        if not self.running and self.app.keyboard_model is None:
            self.model_var.set("Keyboard model pending")
        elif self.app.keyboard_model is None:
            self.model_var.set("Keyboard model unavailable")
        else:
            self.model_var.set("Keyboard model ready")

        if not self.app_lock.acquire(blocking=False):
            return

        try:
            keyboard_locked = self.app.keyboard.locked
            has_manual_points = bool(self.app.manual_corner_points)
            has_detected_corners = self.app.keyboard.corners is not None
            next_corner_label = self.app.get_next_manual_corner_label()
            typed_text = self.app.hand_tracker.text
        finally:
            self.app_lock.release()

        if keyboard_locked:
            calibration_text = "Keyboard locked"
        elif has_manual_points:
            calibration_text = f"Select {next_corner_label}"
        elif has_detected_corners:
            calibration_text = "Detected corners ready"
        else:
            calibration_text = f"Select {next_corner_label}"

        self.calibration_var.set(calibration_text)
        if typed_text != self.last_text:
            self.set_text_box(typed_text)

    def set_text_box(self, text):
        body = text if text else ""
        self.last_text = body
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", body)
        self.text_box.configure(state="disabled")

    def close(self):
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
