Get `data.json`
```
gunzip -c experiments/2025-05-23_13-40-39/1c19b2c_zoo_environment_algorithm/l2f_sac/0001/steps/000000002000000/trajectories.json.gz > data.json
```


Make into video:
```
ffmpeg -framerate 100 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4
```