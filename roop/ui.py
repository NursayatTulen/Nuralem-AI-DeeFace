import os
import sys
import webbrowser
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_ALL
from typing import Any, Callable, Tuple, Optional
import cv2
from PIL import Image, ImageOps

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.predictor import predict_frame, clear_predictor
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

class CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)
    return ROOT

def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('dark')  # Дизайнды заманауи қара түске қойдым
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title('Nuralem AI DeeFace') # Жаңа атау
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    # Басты тақырып
    title_label = ctk.CTkLabel(root, text='NURALEM AI DEEFACE', font=('Roboto', 24, 'bold'))
    title_label.place(relx=0.1, rely=0.03, relwidth=0.8)

    # Фото мен Видео таңдау аймағы
    source_label = ctk.CTkLabel(root, text='Drop Face Image Here', fg_color=ctk.ThemeManager.theme.get('RoopDropArea').get('fg_color'))
    source_label.place(relx=0.1, rely=0.12, relwidth=0.35, relheight=0.25)
    source_label.drop_target_register(DND_ALL)
    source_label.dnd_bind('<<Drop>>', lambda event: select_source_path(event.data))

    target_label = ctk.CTkLabel(root, text='Drop Target Video Here', fg_color=ctk.ThemeManager.theme.get('RoopDropArea').get('fg_color'))
    target_label.place(relx=0.55, rely=0.12, relwidth=0.35, relheight=0.25)
    target_label.drop_target_register(DND_ALL)
    target_label.dnd_bind('<<Drop>>', lambda event: select_target_path(event.data))

    source_button = ctk.CTkButton(root, text='Select Face', cursor='hand2', command=lambda: select_source_path())
    source_button.place(relx=0.1, rely=0.4, relwidth=0.35, relheight=0.07)

    target_button = ctk.CTkButton(root, text='Select Target', cursor='hand2', command=lambda: select_target_path())
    target_button.place(relx=0.55, rely=0.4, relwidth=0.35, relheight=0.07)

    # Баптаулар (Settings)
    keep_fps_value = ctk.BooleanVar(value=roop.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep target FPS', variable=keep_fps_value, command=lambda: setattr(roop.globals, 'keep_fps', not roop.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.52)

    keep_frames_value = ctk.BooleanVar(value=roop.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep temporary frames', variable=keep_frames_value, command=lambda: setattr(roop.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.58)

    skip_audio_value = ctk.BooleanVar(value=roop.globals.skip_audio)
    skip_audio_switch = ctk.CTkSwitch(root, text='Skip target audio', variable=skip_audio_value, command=lambda: setattr(roop.globals, 'skip_audio', skip_audio_value.get()))
    skip_audio_switch.place(relx=0.55, rely=0.52)

    many_faces_value = ctk.BooleanVar(value=roop.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, command=lambda: setattr(roop.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=0.55, rely=0.58)

    # Басқару батырмалары
    start_button = ctk.CTkButton(root, text='START', fg_color='#2ecc71', hover_color='#27ae60', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.1, rely=0.72, relwidth=0.25, relheight=0.08)

    preview_button = ctk.CTkButton(root, text='PREVIEW', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=0.375, rely=0.72, relwidth=0.25, relheight=0.08)

    stop_button = ctk.CTkButton(root, text='CLOSE', fg_color='#e74c3c', hover_color='#c0392b', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=0.65, rely=0.72, relwidth=0.25, relheight=0.08)

    # Статус
    status_label = ctk.CTkLabel(root, text='System Ready', justify='center', font=('Roboto', 12, 'italic'))
    status_label.place(relx=0.1, rely=0.85, relwidth=0.8)

    # Төменгі жақтағы авторлық белгі
    footer_label = ctk.CTkLabel(root, text='Powered by Nuralem AI', text_color='gray')
    footer_label.place(relx=0.1, rely=0.93, relwidth=0.8)

    return root

# Қалған функциялар (create_preview, select_source_path және т.б.) өзгеріссіз қалады
def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider
    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)
    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)
    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))
    preview.bind('<Up>', lambda event: update_face_reference(1))
    preview.bind('<Down>', lambda event: update_face_reference(-1))
    return preview

def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()

def select_source_path(source_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_SOURCE
    if PREVIEW:
        PREVIEW.withdraw()
    if source_path is None:
        source_path = ctk.filedialog.askopenfilename(title='Select Face Image', initialdir=RECENT_DIRECTORY_SOURCE)
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        image = render_image_preview(roop.globals.source_path, (200, 200))
        source_label.configure(image=image, text=None)
    else:
        roop.globals.source_path = None
        source_label.configure(image=None, text='Invalid Image')

def select_target_path(target_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_TARGET
    if PREVIEW:
        PREVIEW.withdraw()
    clear_face_reference()
    if target_path is None:
        target_path = ctk.filedialog.askopenfilename(title='Select Target Video/Image', initialdir=RECENT_DIRECTORY_TARGET)
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        image = render_image_preview(roop.globals.target_path, (200, 200))
        target_label.configure(image=image, text=None)
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame, text=None)
    else:
        roop.globals.target_path = None
        target_label.configure(image=None, text='Invalid Target')

def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT
    if is_image(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save Image', defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save Video', defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        roop.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(roop.globals.output_path)
        start()

def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)

def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()

def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
        clear_predictor()
    elif roop.globals.source_path and roop.globals.target_path:
        init_preview()
        update_preview(roop.globals.reference_frame_number)
        PREVIEW.deiconify()

def init_preview() -> None:
    PREVIEW.title('Preview Mode')
    if is_video(roop.globals.target_path):
        video_frame_total = get_video_frame_total(roop.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(roop.globals.reference_frame_number)

def update_preview(frame_number: int = 0) -> None:
    if roop.globals.source_path and roop.globals.target_path:
        temp_frame = get_video_frame(roop.globals.target_path, frame_number)
        if predict_frame(temp_frame):
            sys.exit()
        source_face = get_one_face(cv2.imread(roop.globals.source_path))
        if not get_face_reference():
            reference_frame = get_video_frame(roop.globals.target_path, roop.globals.reference_frame_number)
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        else:
            reference_face = get_face_reference()
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            temp_frame = frame_processor.process_frame(source_face, reference_face, temp_frame)
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

def update_face_reference(steps: int) -> None:
    clear_face_reference()
    reference_frame_number = int(preview_slider.get())
    roop.globals.reference_face_position += steps
    roop.globals.reference_frame_number = reference_frame_number
    update_preview(reference_frame_number)

def update_frame(steps: int) -> None:
    frame_number = preview_slider.get() + steps
    preview_slider.set(frame_number)
    update_preview(preview_slider.get())