ffmpeg -framerate 100 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4
