from pydub import AudioSegment
import os
import sys


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def preprocess_audio(input_file, output_file):

    print(f"Loading: {input_file}")

    # Load audio regardless of common input format
    audio = AudioSegment.from_file(input_file)

    print(f"Original:")
    print(f"  Sample rate: {audio.frame_rate} Hz")
    print(f"  Channels: {audio.channels}")
    print(f"  Duration: {len(audio) / 1000:.2f} seconds")

    # Convert to mono
    audio = audio.set_channels(TARGET_CHANNELS)

    # Convert to 16 kHz
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    # Export as WAV
    audio.export(
        output_file,
        format="wav",
        parameters=[
            "-ac", "1",
            "-ar", "16000"
        ]
    )

    print("\nProcessed:")
    print(f"  Sample rate: 16000 Hz")
    print(f"  Channels: 1")
    print(f"  Output: {output_file}")

    return output_file


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage:")
        print("python preprocess.py input_audio output.wav")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    preprocess_audio(input_file, output_file)