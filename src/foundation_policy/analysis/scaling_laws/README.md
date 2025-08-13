```
export DEEPSWEEP_SERVER=10.8.0.1:13338; python3 analysis/scaling_laws/job_list.py | curl -X POST --data-binary @- http://$DEEPSWEEP_SERVER/jobs/fp_scaling_laws00
```


```
DEEPSWEEP_SERVER=http://10.0.0.2:13338 && seq 1 50 | xargs -P 24 -I{} deepsweep fp_scaling_laws00 --script tools/deepsweep/deepsweep-client.sh
```


```
./retrieve_results.sh fp_scaling_laws01
```