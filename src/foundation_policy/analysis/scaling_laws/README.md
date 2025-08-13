```
export DEEPSWEEP_SERVER=10.8.0.1:13338; python3 analysis/scaling_laws/job_list.py | curl -X POST --data-binary @- http://$DEEPSWEEP_SERVER/jobs/fp_scaling_laws00
```