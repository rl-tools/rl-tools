set -e
SERVER="localhost:13338"
JOB_ID="cartpole_sweep"
while true; do
  task=$(curl --fail -X POST $SERVER/jobs/$JOB_ID/tasks)
  task_id=$(echo $task | jq -r '.task_id')
  echo $task $task_id
  echo "Do work..."
  sleep 2
  result='{"return": 300}'
  curl -X POST -H 'Content-Type: application/json' --data "$result" $SERVER/jobs/$JOB_ID/tasks/$task_id
done