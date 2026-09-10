import os
import glob
import subprocess
import tempfile

import pyarrow.parquet as pq


# ==========================================
# SETTINGS
# ==========================================

INPUT_DIR = "indictts_deepfake/data"

OUTPUT_DIR = "Ai_training_data/synthetic_100k_wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# FIND PARQUET FILES
# ==========================================

parquet_files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.parquet")
    )
)

print("======================================")
print("HUMAN AUDIO CONVERTER")
print("======================================")

print(f"Found {len(parquet_files)} Parquet files.")

if len(parquet_files) == 0:
    raise FileNotFoundError(
        f"No Parquet files found in {INPUT_DIR}"
    )


# ==========================================
# COUNTERS
# ==========================================

processed = 0
failed = 0


# ==========================================
# PROCESS EACH PARQUET FILE
# ==========================================

for parquet_number, parquet_file in enumerate(
    parquet_files,
    start=1
):

    print("\n--------------------------------------")
    print(
        f"Reading Parquet {parquet_number}/"
        f"{len(parquet_files)}"
    )
    print(
        os.path.basename(parquet_file)
    )
    print("--------------------------------------")


    # Read the Parquet file
    table = pq.read_table(parquet_file)

    rows = table.to_pylist()

    print(f"Rows in file: {len(rows):,}")


    # ======================================
    # PROCESS EACH ROW
    # ======================================

    for row in rows:

        try:

            audio = row["audio"]

            # --------------------------------
            # Get audio bytes
            # --------------------------------

            audio_bytes = audio.get("bytes")

            audio_path = audio.get("path")


            # --------------------------------
            # Case 1: audio stored as bytes
            # --------------------------------

            if audio_bytes is not None:

                with tempfile.NamedTemporaryFile(
                    suffix=".audio",
                    delete=False
                ) as temp_file:

                    temp_file.write(audio_bytes)

                    temp_input = temp_file.name


                # --------------------------------
                # Output filename
                # --------------------------------

                output_filename = (
                    f"human_{processed + 1:06d}.wav"
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    output_filename
                )


                # --------------------------------
                # FFmpeg conversion
                # --------------------------------

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_input,

                        # MONO
                        "-ac",
                        "1",

                        # 16 kHz
                        "-ar",
                        "16000",

                        # 16-bit PCM WAV
                        "-c:a",
                        "pcm_s16le",

                        output_path
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )


                # Delete temporary file
                os.remove(temp_input)


            # --------------------------------
            # Case 2: audio has a path
            # --------------------------------

            elif audio_path is not None:

                output_filename = (
                    f"human_{processed + 1:06d}.wav"
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    output_filename
                )


                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        audio_path,

                        "-ac",
                        "1",

                        "-ar",
                        "16000",

                        "-c:a",
                        "pcm_s16le",

                        output_path
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )


            else:

                raise ValueError(
                    "Audio contains neither bytes nor path"
                )


            # ==================================
            # SUCCESS
            # ==================================

            processed += 1


            if processed % 100 == 0:

                print(
                    f"Converted: {processed:,} | "
                    f"Failed: {failed:,}"
                )


        except Exception as e:

            failed += 1

            print(
                f"ERROR at clip "
                f"{processed + failed}: {e}"
            )


# ==========================================
# DONE
# ==========================================

print("\n======================================")
print("CONVERSION COMPLETE")
print("======================================")

print(
    f"Successfully converted: {processed:,}"
)

print(
    f"Failed:                 {failed:,}"
)

print("\nWAV files are in:")

print(OUTPUT_DIR)