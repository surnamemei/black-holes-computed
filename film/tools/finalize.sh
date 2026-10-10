#!/usr/bin/env bash
# Mux the assembled video with the mixed audio and soft subtitles.
set -euo pipefail
cd "$(dirname "$0")/.."
ffmpeg -y -loglevel error -i out/film_video.mp4 -i out/audio/mix.wav -i out/film.srt \
  -map 0:v -map 1:a -map 2:s -c:v copy -c:a aac -b:a 256k -ar 48000 -c:s mov_text \
  -metadata title="How to Photograph a Black Hole" -metadata:s:s:0 language=eng \
  -metadata comment="Every frame computed: Kerr ray tracing, simulated EHT. Made by Claude (Anthropic), Oct 2026." \
  -movflags +faststart out/how_to_photograph_a_black_hole.mp4
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height -of compact out/how_to_photograph_a_black_hole.mp4
