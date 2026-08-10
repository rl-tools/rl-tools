```
conda create -n habitat python=3.9
conda activate habitat
conda install habitat-sim withbullet -c conda-forge -c aihabitat
python3 render_pose.py --output habitat_golden_poses
```
