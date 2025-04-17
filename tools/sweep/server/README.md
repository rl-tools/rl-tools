
## Create Job
```
cat <<EOF > tasks.ndjson                                                          
{"seed": 0}
{"seed": 1}
{"seed": 2}
EOF
```

```
curl -X POST --data-binary @tasks.ndjson http://localhost:8000/jobs/cartpole_sweep
```


## Take Task
```
curl -X POST http://localhost:8000/jobs/cartpole_sweep/taskscartpole_sweep/tasks
```

## Submit Result
```
curl -X POST -H 'Content-Type: application/json' --data '{"reward": 495.0}' http://localhost:8000/jobs/cartpole_sweep/tasks/1
```
