import ffmpeg
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

class VideoAssembler:
    def __init__(self, fps: int = 30):
        self.fps = fps

    def assemble_video(
        self,
        slide_images: List[Path],
        slide_audios: List[Path],
        output_path: Path,
        vtt_path: Optional[Path] = None,
        chapters_path: Optional[Path] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Path:
        """Combine slide images and audio files into a final MP4 video using FFmpeg."""
        if len(slide_images) != len(slide_audios):
            raise ValueError("Number of images and audio files must match.")

        # Create a list of clips (image + audio)
        clips = []
        for img, audio in zip(slide_images, slide_audios):
            # Input image stream set to duration of audio
            # We use 'loop=1' to repeat the image for the duration of the audio
            # We need to get audio duration first
            probe = ffmpeg.probe(str(audio))
            duration = float(probe['format']['duration'])
            
            v_input = ffmpeg.input(str(img), loop=1, t=duration)
            a_input = ffmpeg.input(str(audio))
            
            # Combine them
            clips.append(v_input)
            clips.append(a_input)

        # Concatenate all clips
        # ffmpeg.concat takes alternating video and audio streams
        joined = ffmpeg.concat(*clips, v=1, a=1).node
        video = joined[0]
        audio = joined[1]

        # Build output command
        output_args = {
            'vcodec': 'libx264',
            'pix_fmt': 'yuv420p',
            'acodec': 'aac',
            'strict': 'experimental',
            'r': self.fps,
            'movflags': '+faststart'
        }

        # Add metadata if provided
        if metadata:
            output_args['metadata'] = [f"{k}={v}" for k, v in metadata.items()]

        # Run FFmpeg
        stream = ffmpeg.output(video, audio, str(output_path), **output_args)
        
        # Overwrite if exists
        stream = ffmpeg.overwrite_output(stream)
        
        try:
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print(f"FFmpeg Error: {e.stderr.decode()}")
            raise e

        # If subtitles are provided, we can either burn them in or include them as a track
        # Here we assume we want them next to the file or embedded.
        # Direct embedding of VTT is tricky in MP4, typically SRT is used or VTT is kept external.
        # We'll stick to keeping them external as VTT for web players, but we can also use FFmpeg to add them.
        
        return output_path

    def generate_subtitles_vtt(self, slide_scripts: List[str], slide_durations: List[float], headings: List[str], output_path: Path):
        """Generate a WebVTT subtitle file."""
        lines = ["WEBVTT\n"]
        current_time = 0.0
        
        for i, (script, duration, heading) in enumerate(zip(slide_scripts, slide_durations, headings)):
            start = self._format_timestamp(current_time)
            end = self._format_timestamp(current_time + duration)
            
            lines.append(f"{start} --> {end}")
            lines.append(f"[{heading}] {script.strip()}\n")
            
            current_time += duration
            
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
