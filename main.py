####################################################################
#                          RO'A NAFI                               #
#                        MANAR SHAWAHNI                            #
#                        ZAINAB JARADAT                            #
####################################################################


import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import spectrogram, butter, lfilter

# Frequency mappings for each character
frequency_map = {
    'a': (100, 1100, 2500),
    'b': (100, 1100, 3000),
    'c': (100, 1100, 3500),
    'd': (100, 1300, 2500),
    'e': (100, 1300, 3000),
    'f': (100, 1300, 3500),
    'g': (100, 1500, 2500),
    'h': (100, 1500, 3000),
    'i': (100, 1500, 3500),
    'j': (300, 1100, 2500),
    'k': (300, 1100, 3000),
    'l': (300, 1100, 3500),
    'm': (300, 1300, 2500),
    'n': (300, 1300, 3000),
    'o': (300, 1300, 3500),
    'p': (300, 1500, 2500),
    'q': (300, 1500, 3000),
    'r': (300, 1500, 3500),
    's': (500, 1100, 2500),
    't': (500, 1100, 3000),
    'u': (500, 1100, 3500),
    'v': (500, 1300, 2500),
    'w': (500, 1300, 3000),
    'x': (500, 1300, 3500),
    'y': (500, 1500, 2500),
    'z': (500, 1500, 3000),
    ' ': (500, 1500, 3500)
}

# Sample rate and signal duration
sample_rate = 44100  # Hz
duration = 0.04  # seconds

def generate_signal(character, frequencies, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sum([np.sin(2 * np.pi * freq * t) for freq in frequencies], axis=0)
    return signal

def encode_string(string, frequency_map, sample_rate, duration):
    encoded_signal = np.array([]) #array of array
    for char in string:
        if char in frequency_map:
            encoded_signal = np.concatenate([encoded_signal, generate_signal(char, frequency_map[char], sample_rate, duration)])
            print(encoded_signal)
        else:
            print(f"Warning: Character '{char}' not in frequency map. Skipping.")

    return encoded_signal

def play_signal(signal):
    sd.play(signal, sample_rate)

def save_signal(signal, file_path):
    sf.write(file_path, signal, sample_rate)

def decode_signal_fft(signal, frequency_map, sample_rate, duration):
    decoded_string = ""
    segment_length = int(sample_rate * duration)
    nperseg_value = min(segment_length, 4096)  # Adjusted nperseg

    for i in range(0, len(signal), segment_length):
        segment = signal[i:i + segment_length]
        f, _, Sxx = spectrogram(segment, fs=sample_rate, nperseg=nperseg_value)
        dominant_freqs = sorted(f[np.argsort(np.amax(Sxx, axis=1))[-3:]])

        closest_char = None
        min_diff = float('inf')
        for char, freqs in frequency_map.items():
            diff = sum(min(abs(freq - char_freq) for char_freq in freqs) for freq in dominant_freqs)
            if diff < min_diff:
                min_diff = diff
                closest_char = char

        decoded_string += closest_char if closest_char else ' '

    return decoded_string

def bandpass_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    b, a = butter(3, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

def decode_signal_filter(signal, frequency_map, sample_rate, duration):
    decoded_string = ""
    for i in range(0, len(signal), int(sample_rate * duration)):
        segment = signal[i:i + int(sample_rate * duration)]
        char_strengths = {}
        for char, freqs in frequency_map.items():
            strength = sum(np.sum(np.abs(bandpass_filter(segment, f - 25, f + 25, sample_rate))) for f in freqs)
            char_strengths[char] = strength
        decoded_string += max(char_strengths, key=char_strengths.get)
    return decoded_string

class SignalEncoderDecoderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Encoder and Decoder")
        self.root.configure(bg='#8E5F92')

        # Main frame that fills the root window
        main_frame = tk.Frame(root, bg='#8E5F92')
        main_frame.pack(fill='both', expand=True)

        # Title and subtitle labels
        title_label = tk.Label(main_frame, text="Signal Encoder and Decoder",font=("Garamond", 26, 'bold'), bg='#8E5F92', fg='white')
        title_label.pack()



        subtitle_label = tk.Label(main_frame, text="Zainab Jaradat | Ro'a Nafi | Manar Shawahni",font=("Garamond", 13, 'bold'), bg='#8E5F92', fg='white')
        subtitle_label.pack()

        # Entry widget for input
        self.input_string_var = tk.StringVar()
        self.input_entry = ttk.Entry(main_frame, textvariable=self.input_string_var, width=50)
        self.input_entry.pack(pady=10)

        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg='#8E5F92')
        buttons_frame.pack()

        style = ttk.Style()
        style.configure('Bold.TButton', font=('Garamond', 12))

        # Button widgets
        ttk.Button(buttons_frame, text="Encode and Play", style='Bold.TButton', command=self.encode_and_play).pack(
            side='left', padx=5, pady=10)
        ttk.Button(buttons_frame, text="Encode and Save", style='Bold.TButton', command=self.encode_and_save).pack(
            side='left', padx=5, pady=10)
        ttk.Button(buttons_frame, text="Decode using FFT", style='Bold.TButton', command=self.decode_fft).pack(
            side='left', padx=5, pady=10)
        ttk.Button(buttons_frame, text="Decode using Filter", style='Bold.TButton', command=self.decode_filter).pack(
            side='left', padx=5, pady=10)
        ttk.Button(buttons_frame, text="Clear", style='Bold.TButton', command=self.clear_text).pack(side='left', padx=5,
                                                                                                    pady=10)
        ttk.Button(buttons_frame, text="Close", style='Bold.TButton', command=root.destroy).pack(side='left', padx=5,
                                                                                                 pady=10)

        # ScrolledText widget for results
        self.result_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, bg='#F0EDE8', fg='black')
        self.result_text.pack(fill='both', expand=True, padx=10, pady=10)

    def encode_and_play(self):
        input_string = self.input_string_var.get()
        encoded_signal = encode_string(input_string, frequency_map, sample_rate, duration)
        play_signal(encoded_signal)

    def encode_and_save(self):
        input_string = self.input_string_var.get()
        encoded_signal = encode_string(input_string, frequency_map, sample_rate, duration)
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave files", "*.wav")])
        if file_path:
            save_signal(encoded_signal, file_path)

    def decode_fft(self):
        file_path = filedialog.askopenfilename(title="Select a WAV file", filetypes=[("Wave files", "*.wav")])
        if file_path:
            signal, _ = sf.read(file_path, dtype=np.float32)
            decoded_string_fft = decode_signal_fft(signal, frequency_map, sample_rate, duration)
            self.result_text.insert(tk.END, f"Decoded (FFT): \n{decoded_string_fft}\n")

    def decode_filter(self):
        file_path = filedialog.askopenfilename(title="Select a WAV file", filetypes=[("Wave files", "*.wav")])
        if file_path:
            signal, _ = sf.read(file_path, dtype=np.float32)
            decoded_string_filter = decode_signal_filter(signal, frequency_map, sample_rate, duration)
            self.result_text.insert(tk.END, f"Decoded (Filter): \n{decoded_string_filter}\n")

    def clear_text(self):
        self.result_text.delete('1.0', tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalEncoderDecoderGUI(root)
    root.mainloop()